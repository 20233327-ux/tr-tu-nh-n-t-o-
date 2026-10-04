import sys

def fix_file(filename, start_marker):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    out_lines = []
    in_main = False
    
    # find the line index to start
    start_idx = -1
    for i, line in enumerate(lines):
        if start_marker in line:
            start_idx = i - 1 # include the print("=" * 70) above it
            break
            
    if start_idx == -1:
        print(f"Could not find marker in {filename}")
        return
        
    for i, line in enumerate(lines):
        if i == start_idx:
            out_lines.append('if __name__ == "__main__":\n')
            in_main = True
            
        if in_main:
            if line.strip() == '':
                out_lines.append('\n')
            else:
                out_lines.append('    ' + line)
        else:
            out_lines.append(line)
            
    with open(filename, 'w', encoding='utf-8') as f:
        f.writelines(out_lines)
    print(f"Fixed {filename}")

fix_file('01_eda.py', 'print("PHASE 1: PHÂN TÍCH DỮ LIỆU THĂM DÒ (EDA)")')
fix_file('02_preprocessing.py', 'print("PHASE 2: TIỀN XỬ LÝ DỮ LIỆU")')
fix_file('03_model_training.py', 'print("PHASE 3: HUẤN LUYỆN & ĐÁNH GIÁ MÔ HÌNH")')
fix_file('04_feature_importance.py', 'print("PHASE 4: PHÂN TÍCH YẾU TỐ ẢNH HƯỞNG")')
fix_file('05_report.py', 'print("PHASE 5: TỔNG HỢP BÁO CÁO")')
