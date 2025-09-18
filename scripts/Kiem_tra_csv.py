import pandas as pd

# Đọc file CSV với encoding utf-8 để xử lý đúng ký tự tiếng Việt
try:
    df = pd.read_csv('E:/HDBank_Hackathon/source/data/raw/csv/final_sua_mapped_v2.csv', encoding='utf-8')
except FileNotFoundError:
    print("Không tìm thấy file 'final_sua_mapped_v2.csv'. Vui lòng kiểm tra đường dẫn file.")
    exit()

# Kiểm tra xem cột 'response' có tồn tại không
if 'response' not in df.columns:
    print("Cột 'response' không tồn tại trong file CSV.")
    exit()

# Hàm đếm tokens (tách từ bằng khoảng trắng)
def count_tokens(text):
    if pd.isna(text):  # Kiểm tra giá trị NaN
        return 0
    return len(str(text).split())

# Tạo cột mới để lưu số lượng tokens
df['token_count'] = df['response'].apply(count_tokens)

# Tìm dòng có số tokens lớn nhất
max_token_row = df['token_count'].idxmax()
max_tokens = df['token_count'].max()
response_content = df.loc[max_token_row, 'response']

# In kết quả
print(f"Ô có số tokens nhiều nhất nằm ở dòng (index): {max_token_row}")
print(f"Số lượng tokens: {max_tokens}")
print(f"Nội dung của ô:\n{response_content}")