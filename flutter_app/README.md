# HDBank AI Chatbot

Ứng dụng chatbot thông minh cho lĩnh vực ngân hàng, được phát triển bằng Flutter.

## Tính năng chính

- 🤖 **AI Chatbot thông minh**: Trả lời các câu hỏi về dịch vụ ngân hàng
- 💳 **Tư vấn thẻ tín dụng**: Thông tin chi tiết về các loại thẻ, hạn mức, phí
- 🏦 **Dịch vụ ngân hàng**: Hướng dẫn các thủ tục, quy định
- ⚡ **Trả lời nhanh**: Gợi ý câu hỏi phổ biến
- 📱 **Giao diện thân thiện**: Thiết kế hiện đại, dễ sử dụng

## Cấu trúc dự án

```
lib/
├── main.dart                 # Entry point
├── models/                   # Data models
│   └── message.dart
├── providers/                # State management
│   └── chat_provider.dart
├── screens/                  # UI screens
│   └── chat_screen.dart
├── services/                 # Business logic
│   └── chat_service.dart
├── utils/                    # Utilities
│   ├── app_theme.dart
│   └── constants.dart
└── widgets/                  # Reusable widgets
    ├── message_bubble.dart
    ├── typing_indicator.dart
    └── quick_replies.dart
```

## Cài đặt và chạy

1. **Cài đặt dependencies:**
   ```bash
   flutter pub get
   ```

2. **Chạy ứng dụng:**
   ```bash
   flutter run
   ```

## Tính năng chatbot

### Các chủ đề được hỗ trợ:
- Thẻ tín dụng và thẻ ghi nợ
- Hạn mức và lãi suất
- Phí dịch vụ
- Quy trình làm thẻ mới
- Xử lý khi mất thẻ
- Thanh toán và rút tiền
- Các quy định của HDBank

### Gợi ý câu hỏi nhanh:
- "Thẻ tín dụng là gì?"
- "Hạn mức thẻ như thế nào?"
- "Cách làm thẻ mới?"
- "Phí thường niên bao nhiêu?"
- "Làm sao khi mất thẻ?"

## Công nghệ sử dụng

- **Flutter**: Framework phát triển ứng dụng
- **Provider**: State management
- **Material Design**: UI/UX design system
- **HTTP**: API communication
- **SharedPreferences**: Local storage

## Tích hợp API

Ứng dụng có thể tích hợp với backend API để:
- Gửi và nhận tin nhắn từ AI model
- Lưu trữ lịch sử chat
- Cập nhật thông tin ngân hàng real-time

## Phát triển tiếp

- [ ] Tích hợp với AI model thực tế
- [ ] Thêm xác thực người dùng
- [ ] Lưu trữ lịch sử chat
- [ ] Hỗ trợ đa ngôn ngữ
- [ ] Push notifications
- [ ] Voice chat support