import 'package:flutter/material.dart';
import '../models/message.dart';
import '../services/chat_service.dart';

class ChatProvider extends ChangeNotifier {
  final List<Message> _messages = [];
  final ChatService _chatService = ChatService();
  bool _isTyping = false;

  List<Message> get messages => _messages;
  bool get isTyping => _isTyping;

  ChatProvider() {
    _initializeChat();
  }

  void _initializeChat() {
    final welcomeMessage = Message(
      id: DateTime.now().millisecondsSinceEpoch.toString(),
      content: 'Xin chào! Tôi là HDBank AI Assistant. Tôi có thể giúp bạn về:\n\n• Thông tin thẻ tín dụng\n• Dịch vụ ngân hàng\n• Hạn mức và phí\n• Quy định và điều khoản\n\nBạn cần hỗ trợ gì hôm nay?',
      isUser: false,
      timestamp: DateTime.now(),
    );
    _messages.add(welcomeMessage);
  }

  Future<void> sendMessage(String content) async {
    if (content.trim().isEmpty) return;

    final userMessage = Message(
      id: DateTime.now().millisecondsSinceEpoch.toString(),
      content: content,
      isUser: true,
      timestamp: DateTime.now(),
    );

    _messages.add(userMessage);
    notifyListeners();

    _setTyping(true);

    try {
      final response = await _chatService.sendMessage(content);
      
      final botMessage = Message(
        id: DateTime.now().millisecondsSinceEpoch.toString(),
        content: response,
        isUser: false,
        timestamp: DateTime.now(),
      );

      _messages.add(botMessage);
    } catch (e) {
      final errorMessage = Message(
        id: DateTime.now().millisecondsSinceEpoch.toString(),
        content: 'Xin lỗi, tôi gặp sự cố kỹ thuật. Vui lòng thử lại sau.',
        isUser: false,
        timestamp: DateTime.now(),
      );
      _messages.add(errorMessage);
    } finally {
      _setTyping(false);
    }
  }

  void _setTyping(bool typing) {
    _isTyping = typing;
    notifyListeners();
  }

  void clearChat() {
    _messages.clear();
    _initializeChat();
    notifyListeners();
  }

  List<String> getQuickReplies() {
    return [
      'Thẻ tín dụng là gì?',
      'Hạn mức thẻ như thế nào?',
      'Cách làm thẻ mới?',
      'Phí thường niên bao nhiêu?',
      'Làm sao khi mất thẻ?',
    ];
  }
}