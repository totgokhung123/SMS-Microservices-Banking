import 'dart:math';

class ChatService {
  final Map<String, String> _bankingResponses = {
    'thẻ tín dụng': 'Thẻ tín dụng HDBank cho phép bạn thực hiện giao dịch trong phạm vi hạn mức tín dụng đã được cấp. Thẻ có thời hạn hiệu lực 3 năm và có thể sử dụng trong và ngoài nước.',
    'hạn mức': 'Hạn mức tín dụng thẻ (HMTD) là số tiền tối đa mà bạn được HDBank cho phép sử dụng tại một thời điểm nhất định. Hạn mức được xét duyệt dựa trên thu nhập và khả năng tài chính.',
    'mất thẻ': 'Khi bị mất thẻ, bạn cần:\n1. Thông báo ngay cho HDBank qua tổng đài\n2. Xác nhận bằng văn bản\n3. Cung cấp thông tin chi tiết về sự cố\n4. Làm thủ tục cấp thẻ mới',
    'phí': 'HDBank áp dụng các loại phí theo biểu phí được công bố trên website. Phí thường niên, phí rút tiền mặt, phí chuyển đổi ngoại tệ được tính theo quy định từng thời kỳ.',
    'làm thẻ': 'Để làm thẻ tín dụng HDBank, bạn cần:\n1. CMND/CCCD\n2. Giấy tờ chứng minh thu nhập\n3. Điền đơn đăng ký\n4. Chờ xét duyệt 3-7 ngày làm việc',
    'lãi suất': 'Lãi suất thẻ tín dụng được áp dụng theo biểu lãi suất HDBank ban hành từng thời kỳ, được công bố trên website hoặc tại điểm giao dịch.',
    'thanh toán': 'Bạn có thể thanh toán thẻ tín dụng qua:\n• Internet Banking\n• Mobile Banking\n• ATM HDBank\n• Quầy giao dịch\n• Chuyển khoản từ ngân hàng khác',
    'rút tiền': 'Hạn mức rút tiền mặt ngoại tệ tại nước ngoài tối đa tương đương 30.000.000 VND/ngày hoặc theo quy định của NHNN.',
  };

  Future<String> sendMessage(String message) async {
    await Future.delayed(const Duration(seconds: 1, milliseconds: 500));
    
    final lowerMessage = message.toLowerCase();
    
    for (final keyword in _bankingResponses.keys) {
      if (lowerMessage.contains(keyword)) {
        return _bankingResponses[keyword]!;
      }
    }

    final defaultResponses = [
      'Tôi hiểu bạn đang quan tâm về dịch vụ ngân hàng. Bạn có thể hỏi cụ thể về thẻ tín dụng, hạn mức, phí, hoặc các dịch vụ khác.',
      'Để tôi hỗ trợ bạn tốt hơn, bạn có thể hỏi về: thẻ tín dụng, hạn mức, phí dịch vụ, cách làm thẻ mới, hoặc quy định của HDBank.',
      'Tôi có thể giúp bạn tìm hiểu về các sản phẩm và dịch vụ của HDBank. Bạn muốn biết thông tin gì cụ thể?',
    ];

    return defaultResponses[Random().nextInt(defaultResponses.length)];
  }
}