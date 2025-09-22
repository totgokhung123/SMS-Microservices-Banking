class AppConstants {
  static const String appName = 'HDBank AI Assistant';
  static const String appVersion = '1.0.0';
  
  // API endpoints
  static const String baseUrl = 'http://localhost:8000';
  static const String chatEndpoint = '/api/chat';
  
  // Storage keys
  static const String chatHistoryKey = 'chat_history';
  static const String userPreferencesKey = 'user_preferences';
  
  // Banking categories
  static const List<String> bankingCategories = [
    'Thẻ tín dụng',
    'Thẻ ghi nợ', 
    'Tài khoản thanh toán',
    'Vay vốn',
    'Tiết kiệm',
    'Chuyển tiền',
    'Dịch vụ khác',
  ];
  
  // Quick reply suggestions
  static const List<String> quickReplies = [
    'Thẻ tín dụng là gì?',
    'Hạn mức thẻ như thế nào?',
    'Cách làm thẻ mới?',
    'Phí thường niên bao nhiêu?',
    'Làm sao khi mất thẻ?',
    'Lãi suất hiện tại?',
    'Cách chuyển tiền?',
    'Mở tài khoản online?',
  ];
}