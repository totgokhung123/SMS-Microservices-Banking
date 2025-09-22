import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'screens/chat_screen.dart';
import 'providers/chat_provider.dart';
import 'utils/app_theme.dart';

void main() {
  runApp(const BankingChatbotApp());
}

class BankingChatbotApp extends StatelessWidget {
  const BankingChatbotApp({super.key});

  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider(
      create: (context) => ChatProvider(),
      child: MaterialApp(
        title: 'HDBank AI Assistant',
        theme: AppTheme.lightTheme,
        home: const ChatScreen(),
        debugShowCheckedModeBanner: false,
      ),
    );
  }
}