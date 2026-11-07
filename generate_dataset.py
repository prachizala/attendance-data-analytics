"""
generate_dataset.py
Generates a realistic, synthetic 90-day employee attendance dataset for workforce data analytics
and predictive machine learning (late arrival forecasting and HR policy simulation).
Uses Python standard library: random, datetime, csv, math (Zero third-party dependencies).
"""

import os
import csv
import math
import random
from datetime import date, datetime, timedelta

# Set fixed seed for 100% reproducible data generation
random.seed(42)

DEPARTMENTS = {
    "Engineering": 18,
    "Sales": 12,
    "Operations": 8,
    "Design": 6,
    "HR": 6
}

FIRST_NAMES = [
    "Aarav", "Priya", "Rohan", "Ananya", "Vikram", "Sneha", "Rahul", "Pooja",
    "Aditya", "Neha", "Amit", "Kavita", "Karan", "Divya", "Manish", "Ritu",
    "Suresh", "Meera", "Deepak", "Tanvi", "Nikhil", "Shweta", "Rajesh", "Simran",
    "Varun", "Isha", "Gaurav", "Preeti", "Sanjay", "Anjali", "Harsh", "Swati",
    "Mayank", "Bhavna", "Kunal", "Rachna", "Ashish", "Nidhi", "Sachin", "Pallavi",
    "Alok", "Monika", "Vivek", "Komal", "Tarun", "Payal", "Pranav", "Shruti",
    "Dev", "Rupal"
]

LAST_NAMES = [
    "Patel", "Shah", "Sharma", "Zala", "Verma", "Mehta", "Desai", "Joshi",
    "Chauhan", "Trivedi", "Rathore", "Bhatt", "Dave", "Pandya", "Gohil",
    "Vaghela", "Panchal", "Soni", "Solanki", "Gajjar"
]

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def generate_employees():
    employees = []
    emp_idx = 1
    names_pool = list(FIRST_NAMES)
    random.shuffle(names_pool)
    
    for dept, count in DEPARTMENTS.items():
        for _ in range(count):
            first = names_pool[(emp_idx - 1) % len(names_pool)]
            last = random.choice(LAST_NAMES)
            emp_id = f"EMP{emp_idx:03d}"
            
            # Commute distance in km (Ahmedabad metro, SG Highway & Ring Road distribution)
            commute_dist = round(max(2.0, min(38.0, random.gauss(15.0, 7.5))), 1)
            
            # Punctuality baseline profile (0.55 to 0.98)
            punctuality = round(random.uniform(0.55, 0.98), 2)
            
            employees.append({
                "employee_id": emp_id,
                "name": f"{first} {last}",
                "department": dept,
                "commute_dist_km": commute_dist,
                "punctuality": punctuality
            })
            emp_idx += 1
    return employees

def generate_attendance():
    employees = generate_employees()
    start_date = date(2026, 5, 1)
    end_date = date(2026, 7, 29) # 90 calendar days
    
    records = []
    current_date = start_date
    day_idx = 0
    monsoon_days = {35, 36, 42, 43, 50, 51, 58, 62} # Heavy monsoon rain days in July
    
    while current_date <= end_date:
        # Skip Sundays
        if current_date.weekday() == 6:
            current_date += timedelta(days=1)
            continue
            
        # Alternate Saturdays off (2nd and 4th Saturday)
        day_of_month = current_date.day
        weekday = current_date.weekday()
        if weekday == 5:
            saturday_number = (day_of_month - 1) // 7 + 1
            if saturday_number in (2, 4):
                current_date += timedelta(days=1)
                continue
                
        day_name = current_date.strftime("%A")
        date_str = current_date.strftime("%Y-%m-%d")
        
        is_mon = (weekday == 0)
        is_fri = (weekday == 4)
        is_rain = (day_idx in monsoon_days)
        
        for emp in employees:
            # Baseline absenteeism check (~3.5%)
            if random.random() < 0.035:
                records.append({
                    "employee_id": emp["employee_id"],
                    "employee_name": emp["name"],
                    "department": emp["department"],
                    "date": date_str,
                    "day_of_week": day_name,
                    "scheduled_start": "09:30",
                    "check_in_time": "",
                    "check_out_time": "",
                    "work_hours": 0.0,
                    "late_minutes": 0,
                    "status": "Absent",
                    "commute_dist_km": emp["commute_dist_km"],
                    "punch_device": "N/A",
                    "geofence_verified": False
                })
                continue
                
            dist = emp["commute_dist_km"]
            punct = emp["punctuality"]
            dept = emp["department"]
            dept_factor = {"Engineering": 0.1, "Sales": 0.45, "Operations": -0.35, "Design": 0.05, "HR": -0.15}[dept]
            
            # Real-world traffic transit friction log-odds
            congestion_penalty = (dist / 14.0) * (1.8 if is_mon else 0.4)
            rain_penalty = 1.8 if (is_rain and dist > 10) else (0.7 if is_rain else 0.0)
            
            z_delay = (
                -1.25 +
                0.10 * (dist - 14.0) +
                congestion_penalty +
                rain_penalty +
                dept_factor -
                3.6 * (punct - 0.74)
            )
            prob_delay = sigmoid(z_delay)
            is_delayed = (random.random() < prob_delay)
            
            if not is_delayed:
                # Arrives on time: check-in between 08:50:00 and 09:30:00
                late_min = 0
                arrival_offset_sec = random.randint(-40 * 60, 0)
            else:
                # Calibration: ~32% severe delays (>15 mins), ~68% minor delays (1-15 mins)
                # Severe delays driven by long distance (>18 km), Monday bottlenecks, and transit shocks
                z_severe = (
                    -3.30 +
                    0.28 * (dist - 14.0) +
                    (2.50 if is_mon else 0.0) +
                    (2.20 if is_rain else 0.0) +
                    dept_factor -
                    2.20 * (punct - 0.74)
                )
                prob_severe = sigmoid(z_severe)
                
                if random.random() < prob_severe:
                    # Severe delay (>15 mins): 16 to 75 mins delay
                    expected = 18.0 + (dist * 0.90) + (12.0 if is_mon else 0.0)
                    late_min = max(16, min(75, int(random.gauss(expected, 4.0))))
                else:
                    # Minor delay: 1 to 15 mins delay (morning signal delay, elevator queue, micro-congestion)
                    # Absorbed completely by 15m grace window or flexible shift start
                    late_min = max(1, min(15, int(random.triangular(1, 15, 6))))
                    
                arrival_offset_sec = late_min * 60 + random.randint(0, 59)
                
            base_start = datetime(current_date.year, current_date.month, current_date.day, 9, 30, 0)
            check_in_dt = base_start + timedelta(seconds=arrival_offset_sec)
            
            is_half_day = random.random() < 0.02
            work_hours = round(random.uniform(3.5, 4.4), 2) if is_half_day else round(random.uniform(7.8, 9.6) - (0.025 if is_fri else 0.0), 2)
            check_out_dt = check_in_dt + timedelta(hours=work_hours)
            
            status = "Half-Day" if is_half_day else ("Late" if late_min > 0 else "Present")
            device = "Mobile App (Face + GPS)" if random.random() < 0.82 else "Web Kiosk QR"
            geofence = True if (device.startswith("Mobile") and random.random() < 0.98) else True
            
            records.append({
                "employee_id": emp["employee_id"],
                "employee_name": emp["name"],
                "department": emp["department"],
                "date": date_str,
                "day_of_week": day_name,
                "scheduled_start": "09:30",
                "check_in_time": check_in_dt.strftime("%H:%M:%S"),
                "check_out_time": check_out_dt.strftime("%H:%M:%S"),
                "work_hours": work_hours,
                "late_minutes": late_min,
                "status": status,
                "commute_dist_km": dist,
                "punch_device": device,
                "geofence_verified": geofence
            })
            
        current_date += timedelta(days=1)
        day_idx += 1
        
    return records

def main():
    print("Generating calibrated 90-day workforce attendance dataset (3,550 shifts)...")
    records = generate_attendance()
    print(f"Generated {len(records)} total attendance records across 50 employees.")
    
    out_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "attendance_90days_synthetic.csv")
    
    fieldnames = [
        "employee_id", "employee_name", "department", "date", "day_of_week",
        "scheduled_start", "check_in_time", "check_out_time", "work_hours",
        "late_minutes", "status", "commute_dist_km", "punch_device", "geofence_verified"
    ]
    
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
        
    print(f"Dataset successfully saved to: {out_csv}")
    
    # Summary metrics
    present_cnt = sum(1 for r in records if r["status"] == "Present")
    late_cnt = sum(1 for r in records if r["status"] == "Late")
    severe_late_cnt = sum(1 for r in records if r["late_minutes"] > 15)
    absent_cnt = sum(1 for r in records if r["status"] == "Absent")
    half_cnt = sum(1 for r in records if r["status"] == "Half-Day")
    
    print("\n--- Dataset Distribution ---")
    print(f"Present (On-Time):   {present_cnt} ({present_cnt / len(records) * 100:.1f}%)")
    print(f"Late (>0 mins):       {late_cnt} ({late_cnt / len(records) * 100:.1f}%)")
    print(f"Severe Late (>15m):   {severe_late_cnt} ({severe_late_cnt / len(records) * 100:.1f}%)")
    print(f"Absent:              {absent_cnt} ({absent_cnt / len(records) * 100:.1f}%)")
    print(f"Half-Day:            {half_cnt} ({half_cnt / len(records) * 100:.1f}%)")
    if late_cnt > 0:
        reduction = (late_cnt - severe_late_cnt) / late_cnt * 100
        print(f"Simulated Grace Window (15m) Tardiness Reduction: {reduction:.1f}%")

if __name__ == "__main__":
    main()
