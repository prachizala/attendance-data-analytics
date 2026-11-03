# Changelog & Technical Interview Cheatsheet

This document outlines the rationale, analytical methodologies, and interview Q&A for the **Attendance Data Analytics** project in clear, simple language.

---

## 1. Why This Project Was Created

When interviewing for Python, Data Analytics, or Fresher Software roles, recruiters specifically look for candidates who can:
1. Formulate a business problem statement from raw or synthetic data.
2. Clean messy inputs and perform **feature engineering** (e.g., calculating minute offsets from a datetime string, categorizing continuous variables into meaningful business tiers).
3. Use industry-standard visualization libraries (**matplotlib**, **seaborn**) to communicate findings clearly.
4. Translate statistical metrics ($r = +0.67$, rolling means, percent compositions) into **actionable business recommendations**.

---

## 2. Technical Interview Cheatsheet

### Q: Why did you synthesize the dataset rather than using real employee records?
> **Answer:** "Real workforce attendance records contain sensitive personal identifiable information (PII) like employee home locations and biometric logs. By synthesizing 3,550 records across 50 employees using Python's standard library (`random.seed(42)`), I created a 100% reproducible, privacy-safe, and realistic dataset exhibiting standard workforce phenomena like commute-driven delays and Monday morning traffic spikes."

### Q: What feature engineering did you perform?
> **Answer:** "Two key features:
> 1. `arrival_offset_mins`: Extracted the check-in time string, calculated total minutes from midnight, and subtracted the 09:30 AM shift baseline (570 minutes). This transformed a raw timestamp into a signed numerical metric where negative values represent early arrivals and positive values represent minutes late.
> 2. `tardiness_tier`: Segmented late arrivals into `Minor (<15m)`, `Moderate (15-30m)`, and `Severe (>30m)` to differentiate routine traffic friction from chronic absenteeism."

### Q: What was the most interesting finding from the analysis?
> **Answer:** "Two key discoveries:
> First, over 68% of 'late' punches were within just 10 minutes of the start time, proving that most tardiness is minor traffic noise rather than intentional disrespect for schedule.
> Second, there is a strong positive Pearson correlation ($r = +0.67$) between commute distance and average daily tardiness. Employees living $>20\text{ km}$ away consistently experienced the highest variance in arrival times, suggesting that flexible 15-minute core-hour windows provide immediate productivity gains without penalizing staff who live far away."
