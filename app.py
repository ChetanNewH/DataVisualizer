import os
import glob
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for web servers
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
# from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask import Flask, render_template, request, redirect, url_for, jsonify, session
# import os

app = Flask(__name__)
app.secret_key = "supersecretkey"
# --- App Configuration ---
# app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["CHART_FOLDER"] = "static/charts"

# --- Ensure Folders Exist ---
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["CHART_FOLDER"], exist_ok=True)

# Global dataframe for single-user context

def generate_insights(df, chart_type, col1=None, col2=None):
    insights = []

    try:
        if chart_type == "Histogram" and col1:
            insights.append(f"Mean of '{col1}' is {df[col1].mean():.2f}")
            insights.append(f"Median of '{col1}' is {df[col1].median():.2f}")
            insights.append("Distribution shows data spread and skewness")

        elif chart_type == "Box Plot (Single Column)" and col1:
            insights.append(f"Q1: {df[col1].quantile(0.25):.2f}")
            insights.append(f"Q3: {df[col1].quantile(0.75):.2f}")
            insights.append("Outliers may exist outside whiskers")

        elif chart_type == "Scatter Plot" and col1 and col2:
            corr = df[col1].corr(df[col2])
            insights.append(f"Correlation between '{col1}' and '{col2}' is {corr:.2f}")
            insights.append(
                "Strong relationship detected" if abs(corr) > 0.5 else "Weak or moderate relationship"
            )

        elif chart_type == "Bar Plot (by Value)" and col1 and col2:
            top = df.groupby(col1)[col2].sum().idxmax()
            insights.append(f"'{top}' has the highest total '{col2}'")
            insights.append("Category-wise contribution varies")

        elif chart_type == "Line Plot (Time-Series)" and col1 and col2:
            insights.append(f"'{col2}' shows trend over time")
            insights.append("Useful for growth or decline analysis")

        elif chart_type == "Correlation Heatmap":
            insights.append("Shows correlation among numeric variables")
            insights.append("Strong colors indicate strong relationships")

        else:
            insights.append("No insights available for this chart")

    except Exception:
        insights.append("Unable to compute insights")

    return insights


df = None  

# --- Routes ---
@app.route("/", methods=["GET", "POST"])
def upload_file():
    """Handles file upload and reads it into the global dataframe."""
    global df
    if request.method == "POST":
        if 'file' not in request.files:
            return render_template("upload.html", error="No file selected.")

        file = request.files['file']
        if file.filename == '':
            return render_template("upload.html", error="No file selected.")

        upload_folder = app.config.get("UPLOAD_FOLDER", os.path.join(os.getcwd(), "uploads"))
        os.makedirs(upload_folder, exist_ok=True)
        filepath = os.path.join(upload_folder, file.filename)
        file.save(filepath)

        try:
            if file.filename.lower().endswith(".csv"):
                df = pd.read_csv(filepath)
            elif file.filename.lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(filepath)
            else:
                return render_template("upload.html", error="Unsupported file format.")
        except Exception as e:
            return render_template("upload.html", error=f"Error reading file: {e}")

        # ⚡ Store the uploaded filename in session
        session['uploaded_file'] = file.filename

        # Redirect to dataclean page after upload
        return redirect(url_for("dataclean"))

    return render_template("upload.html")

@app.route("/overview")
def overview():
    """
    Renders the Project Overview page with all sections in bullet points.
    """
    return render_template("overview.html")
@app.route("/dataclean")
def dataclean():
    """Display first few rows and column list, then suggest next step to chart suggestions."""
    global df
    if df is None:
        return redirect(url_for("upload_file"))

    all_cols = df.columns.tolist()
    table_html = df.head(15).to_html(classes="table table-striped", index=False)
    return render_template("dataclean.html", table_html=table_html, all_cols=all_cols)

from flask import session, jsonify
import os, glob, pandas as pd, numpy as np, csv

@app.route("/clean_data_ajax", methods=["POST"])
def clean_data_ajax():
    """AJAX endpoint to clean the dataset uploaded in this session."""
    global df
    try:
        # --- 1️⃣ Get upload folder ---
        upload_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
        os.makedirs(upload_folder, exist_ok=True)

        # --- 2️⃣ Use the file uploaded in this session ---
        uploaded_file = session.get('uploaded_file')
        if not uploaded_file:
            raise FileNotFoundError("No file uploaded in this session.")

        filepath = os.path.join(upload_folder, uploaded_file)

        # --- 3️⃣ Load dataset dynamically ---
        if filepath.endswith('.csv'):
            with open(filepath, 'r', encoding='utf-8') as f:
                sample = f.read(1024)
                f.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample)
                    sep = dialect.delimiter
                except:
                    sep = ','
            df = pd.read_csv(filepath, sep=sep, encoding='utf-8', low_memory=False)
        elif filepath.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(filepath)
        else:
            raise ValueError("Unsupported file type! Upload a CSV or Excel file.")

        original_shape = df.shape

        # --- 4️⃣ Cleaning Steps ---
        df.dropna(axis=0, how='all', inplace=True)
        df.dropna(axis=1, how='all', inplace=True)

        df.columns = (
            df.columns.astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r'[^0-9a-zA-Z]+', '_', regex=True)
            .str.replace('_+', '_', regex=True)
            .str.strip('_')
        )

        # Numeric cleaning
        numeric_cols = ['rating', 'reviews', 'installs', 'price']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '').str.replace('+', '').str.replace('$', '')
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col].fillna(df[col].mean(), inplace=True)

        float_cols = df.select_dtypes(include=['float', 'int']).columns
        for col in float_cols:
            df[col].fillna(df[col].mean(), inplace=True)

        # Handle 'size'
        if 'size' in df.columns:
            def size_to_float(x):
                x = str(x).strip().upper()
                if x.endswith('M'):
                    return float(x.replace('M', ''))
                elif x.endswith('G'):
                    return float(x.replace('G', '')) * 1024
                else:
                    return np.nan
            df['size'] = df['size'].apply(size_to_float)
            df['size'].fillna(df['size'].mean(), inplace=True)

        # Categorical columns
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            df[col] = df[col].astype(str).str.strip()
            df[col].replace(['', 'nan', 'NaN', 'None', None], np.nan, inplace=True)
            df[col].fillna('Unknown', inplace=True)

        # Preserve key text columns
        preserve_text_cols = ['app', 'category', 'genres', 'type', 'content_rating', 'current_ver', 'android_ver']
        for col in df.select_dtypes(include='object').columns:
            if col in preserve_text_cols:
                df[col] = df[col].str.replace(r'[\t\n\r]+', '', regex=True)

        # Split 'last_updated'
        if 'last_updated' in df.columns:
            df['last_updated'] = pd.to_datetime(df['last_updated'], errors='coerce')
            df['last_updated_day'] = df['last_updated'].dt.day.fillna(0).astype(int)
            df['last_updated_month'] = df['last_updated'].dt.month.fillna(0).astype(int)
            df['last_updated_year'] = df['last_updated'].dt.year.fillna(0).astype(int)
            df.drop(columns=['last_updated'], inplace=True)

        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)

        # --- 5️⃣ Save cleaned dataset ---
        cleaned_filename = os.path.splitext(os.path.basename(filepath))[0] + "_cleaned.csv"
        cleaned_path = os.path.join(upload_folder, cleaned_filename)
        df.to_csv(cleaned_path, index=False)

        # --- 6️⃣ Return preview and download URL ---
        table_html = df.head(15).to_html(classes="table table-striped", index=False)
        file_url = "/uploads/" + cleaned_filename

        return jsonify(success=True, table_html=table_html, file_url=file_url)

    except Exception as e:
        return jsonify(success=False, error=str(e))


# --- Serve uploads folder dynamically ---
from flask import send_from_directory

@app.route('/uploads/<filename>')
def download_file(filename):
    upload_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
    return send_from_directory(upload_folder, filename, as_attachment=True)

@app.route("/chart_suggestions")
def chart_suggestions():
    """Render page with all possible chart suggestions based on uploaded dataset."""
    global df
    if df is None:
        return redirect(url_for("upload_file"))

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns.tolist()

    suggestions = generate_suggestions(numeric_cols, categorical_cols, datetime_cols)

    return render_template(
        "chart_suggestions.html",
        suggestions=suggestions,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        datetime_cols=datetime_cols
    )


def generate_suggestions(numeric_cols, categorical_cols, datetime_cols):
    """Create all chart suggestions with description and columns info."""
    suggestions = []

    # Single Column
    for col in numeric_cols:
        suggestions.append({
            "name": "Histogram",
            "description": f"Visualize the distribution of numeric column '{col}'",
            "columns": col
        })
        suggestions.append({
            "name": "Box Plot (Single Column)",
            "description": f"Show summary statistics of numeric column '{col}'",
            "columns": col
        })

    for col in categorical_cols:
        suggestions.append({
            "name": "Count Plot",
            "description": f"Frequency count of categories in '{col}'",
            "columns": col
        })
        suggestions.append({
            "name": "Pie Chart",
            "description": f"Proportion of each category in '{col}'",
            "columns": col
        })

    # Two-column suggestions
    for col1 in numeric_cols:
        for col2 in numeric_cols:
            if col1 != col2:
                suggestions.append({
                    "name": "Scatter Plot",
                    "description": f"Show relationship between '{col1}' and '{col2}'",
                    "columns": f"{col1} vs {col2}"
                })

    for cat in categorical_cols:
        for num in numeric_cols:
            suggestions.append({
                "name": "Bar Plot (by Value)",
                "description": f"Sum of '{num}' for each category in '{cat}'",
                "columns": f"{cat} vs {num}"
            })
            suggestions.append({
                "name": "Box Plot (by Category)",
                "description": f"Distribution of '{num}' for each category in '{cat}'",
                "columns": f"{cat} vs {num}"
            })

    for dt in datetime_cols:
        for num in numeric_cols:
            suggestions.append({
                "name": "Line Plot (Time-Series)",
                "description": f"Trend of '{num}' over time column '{dt}'",
                "columns": f"{dt} vs {num}"
            })

    # Correlation heatmap
    if len(numeric_cols) >= 2:
        suggestions.append({
            "name": "Correlation Heatmap",
            "description": "Show correlation between all numeric columns",
            "columns": "All numeric columns"
        })

    return suggestions

@app.route("/plot")
def plot_chart():
    chart_type = request.args.get("chart")
    col1 = request.args.get("col1")
    col2 = request.args.get("col2")  # might be None for single-column charts

    if df is None:
        return "No dataset loaded. Go back to Data Cleaning Dashboard."

    plt.figure(figsize=(8,5))
    sns.set(style="whitegrid")

    try:
        # --- Single column charts ---
        if chart_type == "Histogram" and col1:
            sns.histplot(df[col1].dropna(), kde=True, color="#cb00ff")
        elif chart_type == "Box Plot (Single Column)" and col1:
            sns.boxplot(y=df[col1].dropna(), color="#b83d50")
        elif chart_type == "Count Plot" and col1:
            sns.countplot(y=df[col1].dropna(), palette="Purples_r")
        elif chart_type == "Pie Chart" and col1:
            counts = df[col1].value_counts()
            plt.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=sns.color_palette("Purples"))

        # --- Two-column charts ---
        elif chart_type == "Scatter Plot" and col1 and col2:
            sns.scatterplot(x=df[col1], y=df[col2], color="#cb00ff")
        elif chart_type == "Bar Plot (by Value)" and col1 and col2:
            sns.barplot(x=df[col1], y=df[col2], palette="Purples_r")
        elif chart_type == "Box Plot (by Category)" and col1 and col2:
            sns.boxplot(x=df[col1], y=df[col2], palette="Purples_r")
        elif chart_type == "Line Plot (Time-Series)" and col1 and col2:
            sns.lineplot(x=df[col1], y=df[col2], color="#cb00ff")
        elif chart_type == "Heatmap (Crosstab)" and col1 and col2:
            cross = pd.crosstab(df[col1], df[col2])
            sns.heatmap(cross, cmap="Purples", annot=True, fmt="d")
        elif chart_type == "Correlation Heatmap":
            corr = df.select_dtypes(include="number").corr()
            sns.heatmap(corr, cmap="Purples", annot=True, fmt=".2f")
        else:
            return f"Chart type '{chart_type}' not recognized or invalid columns."

        plt.title(chart_type, fontsize=14, color="#cb00ff")
        plt.tight_layout()

        # Convert chart to Base64 for embedding
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", transparent=True)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode()
        buf.close()
        plt.close()

        return render_template("chart_plot.html", chart_img=img_base64, chart_type=chart_type)

    except Exception as e:
        return f"Error plotting chart: {e}"



@app.route("/generate_chart", methods=["GET", "POST"])
def generate_chart():
    """Show chart selection and auto-generate random default chart on load."""
    global df
    if df is None:
        return redirect(url_for("upload_file"))

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns.tolist()

    chart_url = None
    info_text = None

    if request.method == "POST":
        chart_type = request.form["chart_type"]
        col1 = request.form.get("col1")
        col2 = request.form.get("col2")
        chart_url = generate_and_save_chart(df, chart_type, col1, col2)
        info_text = f"Custom chart: {chart_type}"
    else:
        options = possible_charts(df)
        chart_type, col1, col2 = random.choice(options)
        chart_url = generate_and_save_chart(df, chart_type, col1, col2)
        info_text = f"Default chart (random): {chart_type} ({col1}{', ' + col2 if col2 else ''})"

    return render_template(
        "generate_chart.html",
        all_cols=df.columns.tolist(),
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        datetime_cols=datetime_cols,
        chart_url=chart_url,
        auto_chart_info=info_text
    )


def possible_charts(df):
    """Return list of possible chart tuples (chart_type, col1, col2)."""
    charts = []
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns.tolist()

    # Single-column charts
    for col in numeric_cols:
        charts.append(("Histogram", col, None))
        charts.append(("Box Plot (Single Column)", col, None))
    for col in categorical_cols:
        charts.append(("Count Plot", col, None))
        charts.append(("Pie Chart", col, None))

    # Two-column charts
    for col1 in numeric_cols:
        for col2 in numeric_cols:
            if col1 != col2:
                charts.append(("Scatter Plot", col1, col2))
    for cat in categorical_cols:
        for num in numeric_cols:
            charts.append(("Bar Plot (by Value)", cat, num))
            charts.append(("Box Plot (by Category)", cat, num))
    for dt in datetime_cols:
        for num in numeric_cols:
            charts.append(("Line Plot (Time-Series)", dt, num))

    # Correlation heatmap
    if len(numeric_cols) >= 2:
        charts.append(("Correlation Heatmap", None, None))

    return charts


def generate_and_save_chart(df, chart_type, col1=None, col2=None):
    """Generates chart and saves it to static/charts."""
    chart_filename = "chart.png"
    chart_path = os.path.join(app.config["CHART_FOLDER"], chart_filename)
    plt.figure(figsize=(14, 10))
    sns.set_theme(style="whitegrid")

    try:
        if chart_type == "Histogram" and col1:
            sns.histplot(df[col1], kde=True)
        elif chart_type == "Box Plot (Single Column)" and col1:
            sns.boxplot(y=df[col1])
        elif chart_type == "Count Plot" and col1:
            sns.countplot(x=df[col1], order=df[col1].value_counts().index)
        elif chart_type == "Pie Chart" and col1:
            df[col1].value_counts().plot(kind="pie", autopct="%1.1f%%", startangle=140, fontsize=12)
            plt.ylabel("")
        elif chart_type == "Scatter Plot" and col1 and col2:
            sns.scatterplot(data=df, x=col1, y=col2)
        elif chart_type == "Line Plot (Time-Series)" and col1 and col2:
            sns.lineplot(data=df, x=col1, y=col2)
        elif chart_type == "Bar Plot (by Value)" and col1 and col2:
            sns.barplot(data=df, x=col1, y=col2, estimator=sum)
        elif chart_type == "Box Plot (by Category)" and col1 and col2:
            sns.boxplot(data=df, x=col1, y=col2)
        elif chart_type == "Correlation Heatmap":
            numeric_df = df.select_dtypes(include=["number"])
            sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        else:
            sns.countplot(x=df.iloc[:, 0])

        plt.title(chart_type, fontsize=20, pad=20)
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(chart_path, dpi=200, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print("Chart generation error:", e)
        return None

    return url_for('static', filename=f'charts/{chart_filename}')


# @app.route("/generate_chart_ajax", methods=["POST"])
# def generate_chart_ajax():
#     """AJAX endpoint for chart generation without page reload."""
#     global df
#     if df is None:
#         return {"error": "No data loaded"}, 400

#     try:
#         chart_type = request.json["chart_type"]
#         col1 = request.json.get("col1")
#         col2 = request.json.get("col2")

#         chart_url = generate_and_save_chart(df, chart_type, col1, col2)
#         return jsonify({"chart_url": chart_url, "success": True})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
    


@app.route("/generate_chart_ajax", methods=["POST"])
def generate_chart_ajax():
    global df
    data = request.get_json()

    chart_type = data.get("chart_type")
    col1 = data.get("col1")
    col2 = data.get("col2")

    chart_url = generate_and_save_chart(df, chart_type, col1, col2)
    insights = generate_insights(df, chart_type, col1, col2)

    return jsonify({
        "success": True,
        "chart_url": chart_url,
        "insights": insights
    })



if __name__ == "__main__":
    app.run(debug=False, port=5000, host='0.0.0.0')
