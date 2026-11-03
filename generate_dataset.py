"""
generate_dataset.py
Generates a realistic, synthetic 90-day employee attendance dataset for data analytics.
Zero third-party dependencies (uses Python standard library: random, datetime, csv, math).
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
            
            # Base commute distance in km: log-normal distribution around 12km
            commute_dist = round(max(1.5, min(45.0, random.gauss(14.0, 7.5))), 1)
            
            # Punctuality profile: baseline reliability (0.6 to 0.98)
            punctuality = round(random.uniform(0.65, 0.96), 2)
            
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
    while current_date <= end_date:
        # Skip Sundays
        if current_date.weekday() == 6:
            current_date += timedelta(days=1)
            continue
            
        # Alternate Saturdays off (2nd and 4th Saturday)
        day_of_month = current_date.day
        weekday = current_date.weekday()
        if weekday == 5: # Saturday
            saturday_number = (day_of_month - 1) // 7 + 1
            if saturday_number in (2, 4):
                current_date += timedelta(days=1)
                continue
                
        day_name = current_date.strftime("%A")
        date_str = current_date.strftime("%Y-%m-%d")
        
        # Day-of-week factor: Mondays have slightly higher tardiness, Fridays have slight early exits
        monday_penalty = 0.08 if weekday == 0 else 0.0
        friday_leniency = 0.05 if weekday == 4 else 0.0
        
        for emp in employees:
            # Random absence check (illness, leave) ~ 4% rate
            absent_chance = 0.035 + (0.01 if weekday in (0, 4) else 0.0)
            if random.random() < absent_chance:
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
                
            # Determine check-in time based on punctuality and commute
            is_on_time = random.random() < (emp["punctuality"] - monday_penalty - (emp["commute_dist_km"] * 0.003))
            
            if is_on_time:
                # Arrives between 08:50 and 09:30
                arrival_offset_sec = random.randint(-40 * 60, 0)
                late_min = 0
            else:
                # Arrives late: 1 to 75 minutes late
                # Commute distance increases probability of larger delays
                max_late = int(min(90, 20 + emp["commute_dist_km"] * 1.5))
                late_min = int(random.triangular(1, max_late, 12))
                arrival_offset_sec = late_min * 60 + random.randint(0, 59)
                
            base_start = datetime(current_date.year, current_date.month, current_date.day, 9, 30, 0)
            check_in_dt = base_start + timedelta(seconds=arrival_offset_sec)
            
            # Determine work duration (standard is 8.5 to 9.5 hours)
            # Occasional half day (~2%)
            is_half_day = random.random() < 0.02
            if is_half_day:
                work_duration_hours = round(random.uniform(3.5, 4.4), 2)
            else:
                work_duration_hours = round(random.uniform(7.8, 9.6) - (friday_leniency * 0.5), 2)
                
            check_out_dt = check_in_dt + timedelta(hours=work_duration_hours)
            
            # Determine status
            if is_half_day:
                status = "Half-Day"
            elif late_min > 0:
                status = "Late"
            else:
                status = "Present"
                
            # Device type
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
                "work_hours": work_duration_hours,
                "late_minutes": late_min,
                "status": status,
                "commute_dist_km": emp["commute_dist_km"],
                "punch_device": device,
                "geofence_verified": geofence
            })
            
        current_date += timedelta(days=1)
        
    return records

def main():
    records = generate_attendance()
    out_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "attendance_90days_synthetic.csv")
    
    fieldnames = [
        "employee_id", "employee_name", "department", "date", "day_of_week",
        "scheduled_start", "check_in_time", "check_out_time", "work_hours",
        "late_minutes", "status", "commute_dist_km", "punch_device", "geofence_verified"
    ]
    
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
        
    print(f"[OK] Generated {len(records)} attendance records across 50 employees into {out_path}")

if __name__ == "__main__":
    main()
