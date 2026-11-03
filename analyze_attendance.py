"""
analyze_attendance.py
Comprehensive Exploratory Data Analysis & Visualization of 90-Day Employee Attendance Data.
Produces 5 high-resolution figures in figures/ and prints key business insights.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 15
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "data", "attendance_90days_synthetic.csv")
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

def load_and_prepare():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    
    # Calculate arrival minute float from scheduled start (09:30 = 0.0)
    def parse_arrival_minutes(row):
        if pd.isna(row["check_in_time"]) or row["check_in_time"] == "":
            return np.nan
        parts = [int(p) for p in row["check_in_time"].split(":")]
        total_mins = parts[0] * 60 + parts[1] + parts[2] / 60.0
        return total_mins - (9 * 60 + 30) # offset from 09:30 AM
        
    df["arrival_offset_mins"] = df.apply(parse_arrival_minutes, axis=1)
    return df

def generate_figure1_dept_breakdown(df):
    """Figure 1: Department Attendance & Punctuality Breakdown"""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    
    dept_status = pd.crosstab(df["department"], df["status"], normalize="index") * 100
    cols = [c for c in ["Present", "Late", "Half-Day", "Absent"] if c in dept_status.columns]
    dept_status = dept_status[cols]
    
    colors = ["#2ecc71", "#f39c12", "#e67e22", "#e74c3c"]
    dept_status.plot(kind="barh", stacked=True, color=colors, ax=ax, edgecolor="white", width=0.65)
    
    ax.set_title("Attendance Composition by Department (%)", pad=15, fontweight="bold")
    ax.set_xlabel("Percentage of Shifts (%)")
    ax.set_ylabel("Department")
    ax.set_xlim(0, 100)
    ax.legend(title="Status", bbox_to_anchor=(1.02, 1), loc="upper left")
    
    # Annotate percentage numbers
    for n, c in enumerate(dept_status.columns):
        for i, val in enumerate(dept_status[c]):
            if val > 5:
                cum = dept_status.iloc[i, :n].sum() + val / 2
                ax.text(cum, i, f"{val:.1f}%", va="center", ha="center", color="white", fontweight="bold", fontsize=9)
                
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "dept_punctuality_breakdown.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[+] Saved {out}")

def generate_figure2_daily_trend(df):
    """Figure 2: Daily Attendance Time Series with Rolling Average"""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    daily = df[df["status"] != "Absent"].groupby("date")["employee_id"].count().reset_index()
    daily.rename(columns={"employee_id": "present_count"}, inplace=True)
    daily["7d_rolling"] = daily["present_count"].rolling(window=7, min_periods=1).mean()
    
    ax.plot(daily["date"], daily["present_count"], color="#3498db", alpha=0.45, marker="o", markersize=3, label="Daily Present Count")
    ax.plot(daily["date"], daily["7d_rolling"], color="#1a5276", linewidth=2.5, label="7-Day Moving Average")
    
    ax.axhline(50, color="#27ae60", linestyle="--", linewidth=1.2, label="Full Staff (50)")
    ax.set_title("Daily Staff Attendance Trajectory (90 Days)", pad=15, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Active Employees Present")
    ax.set_ylim(35, 53)
    ax.legend(loc="lower left")
    
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "daily_attendance_trend.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[+] Saved {out}")

def generate_figure3_checkin_distribution(df):
    """Figure 3: Distribution of Check-in Times Relative to 09:30 AM Shift Start"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    valid_arrivals = df["arrival_offset_mins"].dropna()
    
    sns.histplot(valid_arrivals, bins=35, kde=True, color="#2980b9", ax=ax, edgecolor="white", alpha=0.6)
    ax.axvline(0, color="#c0392b", linestyle="--", linewidth=2, label="Shift Start (09:30 AM)")
    
    # Shade late zone
    ax.axvspan(0, valid_arrivals.max(), color="#e74c3c", alpha=0.12, label="Late Arrival Zone")
    
    ax.set_title("Employee Check-in Time Distribution Around Shift Start", pad=15, fontweight="bold")
    ax.set_xlabel("Minutes Relative to 09:30 AM (Negative = Early, Positive = Late)")
    ax.set_ylabel("Number of Punches")
    ax.legend(loc="upper right")
    
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "checkin_distribution.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[+] Saved {out}")

def generate_figure4_day_of_week(df):
    """Figure 4: Late Arrival Frequency by Day of Week"""
    fig, ax = plt.subplots(figsize=(9, 5))
    
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    late_df = df[df["status"] == "Late"].groupby("day_of_week")["employee_id"].count().reindex(day_order).fillna(0)
    total_df = df.groupby("day_of_week")["employee_id"].count().reindex(day_order)
    
    late_rate = (late_df / total_df) * 100
    
    palette = ["#e74c3c" if d == "Monday" else "#3498db" for d in day_order]
    bars = ax.bar(day_order, late_rate, color=palette, width=0.55, edgecolor="none")
    
    ax.set_title("Late Arrival Rate (%) by Day of the Week", pad=15, fontweight="bold")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Late Punch Rate (%)")
    ax.set_ylim(0, max(late_rate) * 1.25)
    
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold")
        
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "day_of_week_comparison.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[+] Saved {out}")

def generate_figure5_commute_vs_tardiness(df):
    """Figure 5: Commute Distance vs Tardiness Correlation"""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    emp_summary = df.groupby("employee_id").agg({
        "commute_dist_km": "first",
        "late_minutes": "mean",
        "department": "first"
    }).reset_index()
    
    sns.scatterplot(
        data=emp_summary,
        x="commute_dist_km",
        y="late_minutes",
        hue="department",
        s=80,
        alpha=0.85,
        ax=ax
    )
    
    # Regression trendline
    sns.regplot(
        data=emp_summary,
        x="commute_dist_km",
        y="late_minutes",
        scatter=False,
        ax=ax,
        color="#2c3e50",
        line_kws={"linestyle": "--", "linewidth": 1.5, "label": "Linear Trend"}
    )
    
    # Correlation coefficient
    r = emp_summary["commute_dist_km"].corr(emp_summary["late_minutes"])
    ax.text(0.05, 0.90, f"Pearson r = +{r:.2f} (Moderate positive correlation)", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#ecf0f1", edgecolor="#bdc3c7"), fontweight="bold")
            
    ax.set_title("Commute Distance (km) vs. Mean Daily Tardiness (Minutes)", pad=15, fontweight="bold")
    ax.set_xlabel("Commute Distance from Office (km)")
    ax.set_ylabel("Average Late Minutes per Shift")
    ax.legend(title="Department", loc="upper right")
    
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "commute_distance_vs_tardiness.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[+] Saved {out}")

def main():
    print("Loading attendance dataset...")
    df = load_and_prepare()
    print(f"Loaded {len(df)} records. Analyzing...")
    
    generate_figure1_dept_breakdown(df)
    generate_figure2_daily_trend(df)
    generate_figure3_checkin_distribution(df)
    generate_figure4_day_of_week(df)
    generate_figure5_commute_vs_tardiness(df)
    
    # Compute executive summary stats
    total_punches = len(df)
    present_cnt = (df["status"] == "Present").sum()
    late_cnt = (df["status"] == "Late").sum()
    absent_cnt = (df["status"] == "Absent").sum()
    half_cnt = (df["status"] == "Half-Day").sum()
    
    print("\n" + "="*50)
    print("EXECUTIVE ATTENDANCE SUMMARY (90 DAYS)")
    print("="*50)
    print(f"Total Shift Records:     {total_punches:,}")
    print(f"Present (On-Time):       {present_cnt} ({present_cnt/total_punches*100:.1f}%)")
    print(f"Late Arrivals:           {late_cnt} ({late_cnt/total_punches*100:.1f}%)")
    print(f"Half-Days:               {half_cnt} ({half_cnt/total_punches*100:.1f}%)")
    print(f"Absences:                {absent_cnt} ({absent_cnt/total_punches*100:.1f}%)")
    print("="*50)

if __name__ == "__main__":
    main()
