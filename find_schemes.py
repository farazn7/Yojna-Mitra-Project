import json

with open('data_extraction/schemes.json', encoding='utf-8') as f:
    data = json.load(f)

minority_schemes = [s for s in data if 'minority' in json.dumps(s).lower()]
print(f"--- Minority Schemes ({len(minority_schemes)}) ---")
for s in minority_schemes[:5]:
    print(f"- {s.get('scheme_name')} (Income: {s.get('income_limit', 'None')}, Age: {s.get('age_range', 'None')})")

student_schemes = [s for s in data if 'student' in json.dumps(s).lower() or 'scholarship' in json.dumps(s).lower()]
print(f"\n--- Student/Scholarship Schemes ({len(student_schemes)}) ---")
for s in student_schemes[:5]:
    print(f"- {s.get('scheme_name')}")
    
youth_schemes = [s for s in data if 'unemployed' in json.dumps(s).lower() or 'youth' in json.dumps(s).lower()]
print(f"\n--- Youth/Unemployed Schemes ({len(youth_schemes)}) ---")
for s in youth_schemes[:5]:
    print(f"- {s.get('scheme_name')}")
