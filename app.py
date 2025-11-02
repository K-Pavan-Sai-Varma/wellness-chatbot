# app.py  (Integrated Milestone 3 + Milestone 4) — DB-backed knowledge + embeddings
import os
import sqlite3
import bcrypt
import re
from datetime import datetime, date, timedelta, timezone
from zoneinfo import ZoneInfo
import pytz
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
# ---- ML model imports from Milestone 3 (paths unchanged) ----
from sentence_transformers import SentenceTransformer, util
from transformers import MarianMTModel, MarianTokenizer
from langdetect import detect, DetectorFactory
import torch

# ---------------- Flask Setup ----------------
app = Flask(__name__)
app.secret_key = "supersecretkey"

# ---------------- Database paths ----------------
MASTER_DB = "database.db"       # unified DB
LEGACY_USERS_DB = "users.db"    # your old Milestone3 DB (if present)

# ------------------ DB helpers ------------------
def run_query(db_path, q, params=(), fetch=False):
    """Generic helper to run a query against a sqlite db file."""
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(q, params)
        if fetch:
            rows = cur.fetchall()
            return rows
        conn.commit()

def run_master(q, params=(), fetch=False):
    return run_query(MASTER_DB, q, params, fetch)

# ------------------ Initialize / Migrate DB ------------------
def init_db_and_migrate():
    # Create master tables
    run_master('''CREATE TABLE IF NOT EXISTS users(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE,
                    pwd BLOB,
                    name TEXT,
                    age INTEGER,
                    lang TEXT
                 )''')
    run_master('''CREATE TABLE IF NOT EXISTS queries(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    query TEXT,
                    response TEXT,
                    feedback TEXT,
                    comment TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                 )''')
    run_master('''CREATE TABLE IF NOT EXISTS knowledge(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT,
                    subtopic TEXT,
                    tip TEXT
                 )''')
    run_master('''CREATE TABLE IF NOT EXISTS admin(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE,
                    password TEXT
                 )''')

    # Insert default admin if missing
    existing = run_master("SELECT * FROM admin WHERE email=?", ("admin@gmail.com",), True)
    if not existing:
        run_master("INSERT INTO admin (email, password) VALUES (?,?)", ("admin@gmail.com", "admin@123"))

    # Migrate users from legacy users.db if exists and if master users table empty
    if os.path.exists(LEGACY_USERS_DB):
        master_users_rows = run_master("SELECT COUNT(*) FROM users", fetch=True)
        master_users = master_users_rows[0][0] if master_users_rows else 0
        if master_users == 0:
            try:
                with sqlite3.connect(LEGACY_USERS_DB) as conn:
                    cur = conn.cursor()
                    cur.execute("SELECT email, pwd, name, age, lang FROM users")
                    rows = cur.fetchall()
                    for row in rows:
                        email, pwd_blob, name, age, lang = row
                        run_master("INSERT OR IGNORE INTO users (email, pwd, name, age, lang) VALUES (?,?,?,?,?)",
                                   (email, pwd_blob, name, age, lang))
                print("Migrated users from users.db -> database.db")
            except Exception as e:
                print("User migration failed:", e)

# Run initialization
init_db_and_migrate()

# ---------------- Load Models (local paths) ----------------
DetectorFactory.seed = 0
# SentenceTransformer model (use your local path)
st_model = SentenceTransformer("offline_models/sentence_transformers/paraphrase")
# Marian translation (local path)
en2hi_model = MarianMTModel.from_pretrained("offline_models/marianmt/opus-mt-en-hi")
en2hi_tokenizer = MarianTokenizer.from_pretrained("offline_models/marianmt/opus-mt-en-hi")

# ---------------- Translation & Language Detection ----------------
def translate_en_to_hi(text_list):
    if isinstance(text_list, str):
        text_list = [text_list]
    tokens = en2hi_tokenizer(text_list, return_tensors="pt", padding=True, truncation=True)
    translated = en2hi_model.generate(**tokens)
    translations = []
    for orig, trans in zip(text_list, translated):
        translated_text = en2hi_tokenizer.decode(trans, skip_special_tokens=True)
        bullet = "• " if orig.strip().startswith("•") else ""
        translations.append(bullet + translated_text.lstrip("• ").strip())
    return translations

def detect_language_fallback(text):
    try:
        lang = detect(text)
    except:
        lang = "en"
    if re.search(r"[ऀ-ॿ]", text):
        lang = "hi"
    return lang

# ---------------- Knowledge Embeddings (DB-driven) ----------------
# Globals that store the list of subtopics and their embeddings
kb_subtopics = []      # list of subtopic strings (e.g. "fever", "stomach_pain")
kb_embeddings = None   # torch tensor of embeddings corresponding to kb_subtopics
MATCH_THRESHOLD = 0.55

def refresh_kb_embeddings():
    """Read distinct subtopics from the knowledge table and compute embeddings."""
    global kb_subtopics, kb_embeddings
    rows = run_master("SELECT DISTINCT subtopic FROM knowledge WHERE subtopic IS NOT NULL AND TRIM(subtopic)<>''", fetch=True) or []
    kb_subtopics = [r[0] for r in rows] if rows else []
    if kb_subtopics:
        try:
            kb_embeddings = st_model.encode(kb_subtopics, convert_to_tensor=True)
            # Ensure tensor on CPU for consistent behavior
            if hasattr(kb_embeddings, "is_cuda") and kb_embeddings.is_cuda:
                kb_embeddings = kb_embeddings.cpu()
        except Exception as e:
            print("Failed to compute kb embeddings:", e)
            kb_embeddings = None
    else:
        kb_embeddings = None

# initial load
refresh_kb_embeddings()

def find_best_symptom(user_input):
    """Return the best matching subtopic (string) from DB or None."""
    global kb_subtopics, kb_embeddings
    if not kb_subtopics or kb_embeddings is None:
        return None
    try:
        user_embedding = st_model.encode(user_input, convert_to_tensor=True)
        if hasattr(user_embedding, "is_cuda") and user_embedding.is_cuda and not kb_embeddings.is_cuda:
            user_embedding = user_embedding.cpu()
        cosine_scores = util.cos_sim(user_embedding, kb_embeddings)[0]
        best_idx = torch.argmax(cosine_scores).item()
        best_score = float(cosine_scores[best_idx])
        if best_score >= MATCH_THRESHOLD:
            return kb_subtopics[best_idx]
    except Exception as e:
        print("Semantic match failed:", e)
    return None

# ---------------- User helper functions ----------------
def get_user_by_email(email):
    rows = run_master("SELECT id, email, pwd, name, age, lang FROM users WHERE email=?", (email,), True) or []
    if not rows:
        return None
    r = rows[0]
    return {"id": r[0], "email": r[1], "pwd": r[2], "name": r[3], "age": r[4], "lang": r[5]}

def get_user_id_by_email(email):
    rows = run_master("SELECT id FROM users WHERE email=?", (email,), True) or []
    return rows[0][0] if rows else None

# ---------------- Routes (User) ----------------
@app.route('/')
def home_index():
    return render_template("index.html")

@app.route('/register', methods=['GET', 'POST'])
def register():
    # If already logged in, logout first for safety
    if 'user' in session:
        session.pop('user', None)

    if request.method == "POST":
        email = request.form.get("email", "").strip()
        pwd = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")
        if pwd != confirm:
            flash("Passwords do not match", "danger")
            return render_template("auth.html", page="register")
        if len(pwd) < 8 or not any(c.isdigit() for c in pwd) or not any(not c.isalnum() for c in pwd):
            flash("Password must be 8+ characters and include a number & special character", "danger")
            return render_template("auth.html", page="register")
        existing = run_master("SELECT 1 FROM users WHERE email=?", (email,), True) or []
        if existing:
            flash("User already exists. Please log in instead.", "danger")
            return render_template("auth.html", page="register")

        hashed = bcrypt.hashpw(pwd.encode(), bcrypt.gensalt())
        try:
            run_master("INSERT INTO users (email, pwd, name, age, lang) VALUES (?,?,?,?,?)",
                       (email, hashed, None, None, None))
            created = get_user_by_email(email)
            if not created:
                flash("Registration failed (DB). Please try again.", "danger")
                return render_template("auth.html", page="register")
            session.clear()
            session['user'] = email
            return redirect(url_for('profile'))
        except sqlite3.IntegrityError:
            flash("This email is already registered. Please log in.", "danger")
        except Exception as e:
            flash(f"Registration error: {e}", "danger")
    return render_template("auth.html", page="register")


@app.route('/logout')
def logout():
    # Clear all user session data
    session.pop('user', None)
    session.pop('chat_history_tabs', None)
    session.clear()
    flash("You have been logged out successfully.", "info")
    return redirect(url_for('home_index'))


@app.route('/login', methods=['GET','POST'])
def login():
    if 'user' in session:
        u = get_user_by_email(session['user'])
        # if no user found in DB for this session, clear session and proceed to login page
        if not u:
            session.pop('user', None)
            return render_template("auth.html", page="login")
        if not u.get("name") or not u.get("age") or not u.get("lang"):
            return redirect(url_for('profile'))
        return redirect(url_for('chat'))

    if request.method == "POST":
        email = request.form.get("email","").strip()
        pwd = request.form.get("password","")
        user = get_user_by_email(email)
        if not user:
            flash("No user found","danger")
        else:
            stored = user.get("pwd")
            try:
                ok = bcrypt.checkpw(pwd.encode(), stored)
            except Exception:
                ok = (pwd == stored)
            if not ok:
                flash("Wrong password","danger")
            else:
                session['user'] = email
                session['chat_history_tabs'] = session.get('chat_history_tabs', [])
                if not user.get("name") or not user.get("age") or not user.get("lang"):
                    return redirect(url_for('profile'))
                return redirect(url_for('chat'))
    return render_template("auth.html", page="login")

@app.route('/reset', methods=['GET','POST'])
def reset():
    if request.method == "POST":
        email = request.form.get("email","").strip()
        pwd = request.form.get("password","")
        confirm = request.form.get("confirm_password","")
        if pwd != confirm:
            flash("Passwords do not match", "danger")
        else:
            user = get_user_by_email(email)
            if not user:
                flash("No user found with this email", "danger")
            else:
                hashed = bcrypt.hashpw(pwd.encode(), bcrypt.gensalt())
                run_master("UPDATE users SET pwd=? WHERE email=?", (hashed, email))
                flash("Password updated successfully", "success")
                return redirect(url_for('login'))
    return render_template("auth.html", page="reset")

@app.route('/profile', methods=['GET','POST'])
def profile():
    if 'user' not in session:
        return redirect(url_for('login'))
    user = get_user_by_email(session['user'])
    if not user:
        # If user record was deleted (admin deleted all users), clear session and redirect
        session.pop('user', None)
        flash("Your account was not found (maybe deleted). Please register again.", "warning")
        return redirect(url_for('register'))

    if request.method == "POST":
        name = request.form.get("name","").strip()
        age = request.form.get("age")
        lang = request.form.get("lang")
        new_pwd = request.form.get("password","").strip()
        if new_pwd:
            hashed = bcrypt.hashpw(new_pwd.encode(), bcrypt.gensalt())
            run_master("UPDATE users SET pwd=?, name=?, age=?, lang=? WHERE email=?", (hashed, name, age, lang, user["email"]))
        else:
            run_master("UPDATE users SET name=?, age=?, lang=? WHERE email=?", (name, age, lang, user["email"]))
        flash("Profile updated successfully", "success")
        return redirect(url_for('chat'))
    # Pass user tuple data to the template like before
    return render_template("auth.html", page="profile", user=(user.get("email"), user.get("pwd"), user.get("name"), user.get("age"), user.get("lang")))

@app.route('/chat')
def chat():
    if 'user' not in session:
        return redirect(url_for('login'))
    user = get_user_by_email(session['user'])
    if not user or not user.get("name") or not user.get("age") or not user.get("lang"):
        return redirect(url_for('profile'))
    session.setdefault('chat_history_tabs', [])
    return render_template("chat.html", user=user)

@app.route('/send_message', methods=['POST'])
def send_message():
    if 'user' not in session:
        return jsonify({"bot_reply": "Please login first"})
    msg = request.form.get("user_message","").strip()
    if not msg:
        return jsonify({"bot_reply": "Please enter a message"})
    lang = detect_language_fallback(msg)
    msg_lower = msg.lower()
    greetings_en = ["hi", "hello", "hey", "good morning", "good evening"]
    greetings_hi = ["नमस्ते", "नमस्कार"]

    if any(greet in msg_lower for greet in greetings_en):
        reply = "👋 Hello! How are you? How can I help you today?"
    elif any(greet in msg for greet in greetings_hi):
        reply = "👋 नमस्ते! आप कैसे हैं? मैं आपकी कैसे मदद कर सकता हूँ?"
    else:
        best_symptom = find_best_symptom(msg)
        if not best_symptom:
            reply = "❓ माफ़ करें, मैं समझ नहीं पाया।" if lang=="hi" else "❓ Sorry, I didn't understand."
        else:
            # preference: use knowledge DB for tips
            tips_rows = run_master("SELECT tip FROM knowledge WHERE subtopic=? LIMIT 50", (best_symptom,), True) or []
            tips = [r[0] for r in tips_rows] if tips_rows else []

            if lang == "hi":
                tips = translate_en_to_hi(tips)
                symptom_keyword = best_symptom.replace("_", " ")
                translated_keyword = translate_en_to_hi(symptom_keyword)[0]
                symptom_title = f"{translated_keyword} सुझाव"
                disclaimer = '''
                🔴 अस्वीकरण : यहां दी गई जानकारी सामान्य स्वास्थ्य मार्गदर्शन 
                हेतु है और पेशेवर चिकित्सा देखभाल का विकल्प नहीं है। 
                व्यक्तिगत सलाह के लिए योग्य स्वास्थ्य पेशेवर से संपर्क करें'''
            else:
                symptom_title = f"{best_symptom.replace('_',' ').capitalize()} tips"
                disclaimer = '''
                🔴  Disclaimer : The information provided here is for general
                health guidance only and is not a replacement for 
                professional medical care. Please consult a qualified 
                healthcare provider for personalized advice.'''
            reply = f"\n💡 {symptom_title}:\n\n" + "\n".join(tips) + "\n\n"+disclaimer

    # Save query (one-time) to queries table for admin analytics
    user_id = get_user_id_by_email(session.get('user'))
    try:
        run_master("INSERT INTO queries (user_id, query, response) VALUES (?,?,?)", (user_id, msg, reply))
    except Exception as e:
        print("Failed to log query:", e)

    # Maintain session-only chat history (unchanged behavior)
    tabs = session.get('chat_history_tabs', [])
    if not tabs:
        tabs.append([{"role":"You","text":msg},{"role":"Bot","text":reply}])
    else:
        tabs[-1].append({"role":"You","text":msg})
        tabs[-1].append({"role":"Bot","text":reply})
    session['chat_history_tabs'] = tabs

    return jsonify({"bot_reply": reply})

@app.route("/delete_account", methods=["POST"])
def delete_account():
    if 'user' not in session:
        return jsonify({"success": False, "message": "User not logged in"})

    email = session.get("user")
    user = get_user_by_email(email)
    if not user:
        return jsonify({"success": False, "message": "User not found"})

    try:
        # Delete user from DB
        run_master("DELETE FROM users WHERE email = ?", (email,))
        # Clear session
        session.clear()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "message":str(e)})
# ---------------- Admin routes ----------------
@app.route('/admin/login', methods=['GET','POST'])
def admin_login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        row = run_master("SELECT email, password FROM admin WHERE email=?", (email,), True) or []
        if row and row[0][1] == password:
            session['admin'] = email
            return redirect(url_for('admin_dashboard'))
        else:
             flash("Invalid credentials!", "danger")
    return render_template('admin-login.html')

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin', None)
    return redirect(url_for('home_index'))

@app.route('/admin/dashboard')
def admin_dashboard():
    if 'admin' not in session:
        return redirect(url_for('admin_login'))

    total_users_row = run_master("SELECT COUNT(*) FROM users", fetch=True) or [(0,)]
    total_users = total_users_row[0][0]
    total_queries_row = run_master("SELECT COUNT(*) FROM queries", fetch=True) or [(0,)]
    total_queries = total_queries_row[0][0]
    positive_feedback = run_master("SELECT COUNT(*) FROM queries WHERE feedback='positive'", fetch=True) or [(0,)]
    positive_feedback = positive_feedback[0][0]
    negative_feedback = run_master("SELECT COUNT(*) FROM queries WHERE feedback='negative'", fetch=True) or [(0,)]
    negative_feedback = negative_feedback[0][0]

    cat_data = run_master("SELECT category, COUNT(*) FROM knowledge GROUP BY category", fetch=True) or []
    categories = [r[0] for r in cat_data] if cat_data else ["Symptoms", "Prevention", "First Aid", "Wellness"]
    counts = [r[1] for r in cat_data] if cat_data else [1,1,1,1]

    recent_qs = run_master("SELECT q.id, u.email, q.query, q.feedback, q.created_at FROM queries q LEFT JOIN users u ON q.user_id = u.id ORDER BY q.created_at DESC LIMIT 10", fetch=True) or []
    return render_template('admin-dashboard.html',
                           total_users=total_users,
                           total_queries=total_queries,
                           pos_feedback=positive_feedback,
                           neg_feedback=negative_feedback,
                           categories=categories,
                           counts=counts,
                           recent_qs=recent_qs)

@app.route('/admin/users')
def admin_users():
    if 'admin' not in session:
        return redirect(url_for('admin_login'))
    conn = sqlite3.connect('database.db')
    cur = conn.cursor()
    cur.execute("SELECT * FROM users")
    users = cur.fetchall() or []
    conn.close()
    return render_template('admin_users.html', users=users)

@app.route('/admin/knowledge')
def admin_knowledge():
    if 'admin' not in session:
        return redirect(url_for('admin_login'))
    cats = run_master("SELECT DISTINCT category FROM knowledge", fetch=True) or []
    data = run_master("SELECT * FROM knowledge ORDER BY category, subtopic", fetch=True) or []
    return render_template('admin-knowledge.html', categories=cats, data=data)

@app.route('/admin/add_tip', methods=['POST'])
def admin_add_tip():
    if 'admin' not in session:
        return redirect(url_for('admin_login'))

    cat = request.form.get('category')
    sub = request.form.get('subtopic')
    tip = request.form.get('tip')

    if not (cat and sub and tip):
        flash("All fields required", "danger")
        return redirect(url_for('admin_knowledge'))

    # Add bullet points (no extra blank lines)
    tip_lines = [line.strip() for line in tip.split('\n') if line.strip()]
    tip_text = '• ' + '\n• '.join(tip_lines)

    run_master("INSERT INTO knowledge (category, subtopic, tip) VALUES (?,?,?)", (cat, sub, tip_text))
    # refresh embeddings so new subtopic is available immediately
    refresh_kb_embeddings()
    return redirect(url_for('admin_knowledge'))

@app.route('/admin/delete_tip/<int:tid>', methods=['POST'])
def admin_delete_tip(tid):
    if 'admin' not in session:
        return redirect(url_for('admin_login'))
    run_master("DELETE FROM knowledge WHERE id=?", (tid,))
    refresh_kb_embeddings()
    flash("Deleted successfully!", "info")
    return redirect(url_for('admin_knowledge'))

@app.route('/admin/feedback')
def admin_feedback():
    if 'admin' not in session:
        return redirect(url_for('admin_login'))

    feedbacks = run_master("""
        SELECT 
            u.email AS user,
            q.query,
            q.feedback AS type,
            q.comment,
            q.created_at AS time
        FROM queries q
        LEFT JOIN users u ON q.user_id = u.id
        WHERE q.feedback IS NOT NULL OR q.comment IS NOT NULL
        ORDER BY q.created_at DESC
    """, fetch=True) or []

    feedbacks = [
        {
            "user": f[0],
            "query": f[1],
            "type": f[2],
            "comment": f[3],
            "time": (datetime.strptime(f[4], "%Y-%m-%d %H:%M:%S")
                        .replace(tzinfo=timezone.utc)
                        .astimezone(ZoneInfo("Asia/Kolkata"))
                        .strftime("%d-%m-%Y %I:%M:%S %p")) if f[4] else None
        }
        for f in feedbacks
    ]
    return render_template('admin-feedback.html', feedbacks=feedbacks)

@app.route('/admin/settings')
def admin_settings():
    if 'admin' not in session:
        return redirect(url_for('admin_login'))
    return render_template('admin-settings.html')

# ------------------ Admin Delete Actions ------------------

@app.route('/admin/delete_feedbacks', methods=['POST'])
def delete_feedbacks():
    if 'admin' not in session:
        return redirect('/admin/login')

    try:
        con = sqlite3.connect("database.db")
        cur = con.cursor()
        cur.execute("DELETE FROM queries WHERE feedback IS NOT NULL OR comment IS NOT NULL;")
        con.commit()
        con.close()
        flash("✅ All feedback records deleted successfully!", "success")
    except Exception as e:
        flash(f"❌ Error deleting feedbacks: {e}", "danger")

    return redirect('/admin/settings')


@app.route('/admin/delete_users', methods=['POST'])
def delete_users():
    if 'admin' not in session:
        return redirect('/admin/login')

    try:
        con = sqlite3.connect("database.db")
        cur = con.cursor()
        cur.execute("DELETE FROM users;")
        con.commit()
        con.close()
        flash("✅ All user records deleted successfully!", "success")
    except Exception as e:
        flash(f"❌ Error deleting users: {e}", "danger")

    return redirect('/admin/settings')


@app.route('/admin/delete_tips', methods=['POST'])
def delete_tips():
    if 'admin' not in session:
        return redirect('/admin/login')

    try:
        con = sqlite3.connect("database.db")
        cur = con.cursor()
        cur.execute("DELETE FROM knowledge;")
        con.commit()
        con.close()
        flash("✅ All tips and categories deleted successfully!", "success")
    except Exception as e:
        flash(f"❌ Error deleting tips: {e}", "danger")

    return redirect('/admin/settings')


@app.route('/admin/change_password', methods=['POST'])
def admin_change_password():
    if 'admin' not in session:
        return redirect(url_for('admin_login'))
    curr = request.form.get('current_password')
    newp = request.form.get('new_password')
    conf = request.form.get('confirm_password')
    admin_row = run_master("SELECT id, email, password FROM admin WHERE email=?", (session['admin'],), True) or []
    if not admin_row:
        flash("Admin not found", "danger")
        return redirect(url_for('admin_settings'))
    admin = admin_row[0]
    if admin[2] != curr:
        flash("Current password is wrong!", "danger")
        return redirect(url_for('admin_settings'))
    if newp != conf:
        flash("Passwords do not match!", "danger")
        return redirect(url_for('admin_settings'))
    run_master("UPDATE admin SET password=? WHERE email=?", (newp, session['admin']))
    return redirect(url_for('admin_settings'))

# ---------------- Feedback endpoint ----------------
@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    if 'user' not in session:
        return jsonify({"error": "not_logged_in"}), 401

    data = request.get_json() or {}
    fb_type = data.get("feedback")
    comment = (data.get("comment") or "").strip() or None

    if not fb_type and not comment:
        return jsonify({"error": "no_feedback"}), 400

    user_id = get_user_id_by_email(session['user'])
    latest_query = run_master("SELECT id FROM queries WHERE user_id = ? ORDER BY id DESC LIMIT 1", (user_id,), fetch=True) or []
    if not latest_query:
        return jsonify({"error": "no_query_found"}), 400
    qid = latest_query[0][0]

    existing = run_master("SELECT feedback, comment FROM queries WHERE id=?", (qid,), fetch=True) or []
    prev_feedback, prev_comment = existing[0] if existing else (None, None)
    new_feedback = fb_type or prev_feedback
    new_comment = comment or prev_comment

    run_master("UPDATE queries SET feedback = ?, comment = ? WHERE id = ?", (new_feedback, new_comment, qid))
    return jsonify({"success": True})

# ---------------- Admin API (dashboard data) ----------------
from flask import jsonify, session
from datetime import date, timedelta

@app.route('/admin/api/dashboard_data')
def admin_api_dashboard():
    if 'admin' not in session:
        return jsonify({"error": "not_logged_in"}), 401

    total_users = (run_master("SELECT COUNT(*) FROM users", fetch=True) or [(0,)])[0][0]
    total_queries = (run_master("SELECT COUNT(*) FROM queries", fetch=True) or [(0,)])[0][0]

    positive_count = (run_master("SELECT COUNT(*) FROM queries WHERE feedback='positive'", fetch=True) or [(0,)])[0][0]
    negative_count = (run_master("SELECT COUNT(*) FROM queries WHERE feedback='negative'", fetch=True) or [(0,)])[0][0]
    total_feedback = positive_count + negative_count
    positive_feedback_percent = round((positive_count / total_feedback) * 100, 1) if total_feedback > 0 else 0.0

    total_health_topics = (run_master("SELECT COUNT(DISTINCT subtopic) FROM knowledge", fetch=True) or [(0,)])[0][0]

    cat_rows = run_master("SELECT category, COUNT() FROM knowledge GROUP BY category ORDER BY COUNT() DESC", fetch=True) or []
    categories = {r[0]: r[1] for r in cat_rows}

    days_rows = run_master("""
        SELECT DATE(created_at) as d, COUNT(*) 
        FROM queries
        WHERE DATE(created_at) >= DATE('now', '-6 days')
        GROUP BY d ORDER BY d
    """, fetch=True) or []

    labels, values = [], []
    for i in range(6, -1, -1):
        d = (date.today() - timedelta(days=i)).isoformat()
        labels.append(d[-5:])
        found = next((r for r in days_rows if r[0] == d), None)
        values.append(found[1] if found else 0)

    return jsonify({
        "admin_name": session.get('admin'),
        "total_users": total_users,
        "total_queries": total_queries,
        "positive_feedback_percent": positive_feedback_percent,
        "total_health_topics": total_health_topics,
        "categories": categories,
        "daily_labels": labels,
        "daily_queries": values
})


# ---------------- Run App ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000,debug=True)
