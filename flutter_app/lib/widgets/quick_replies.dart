import 'package:flutter/material.dart';
import '../utils/app_theme.dart';

class QuickReplies extends StatelessWidget {
  final List<String> replies;
  final Function(String) onReplyTap;

  const QuickReplies({
    super.key,
    required this.replies,
    required this.onReplyTap,
  });

  @override
  Widget build(BuildContext context) {
    if (replies.isEmpty) return const SizedBox.shrink();

    return Container(
      height: 50,
      padding: const EdgeInsets.symmetric(horizontal: 16),
      child: ListView.builder(
        scrollDirection: Axis.horizontal,
        itemCount: replies.length,
        itemBuilder: (context, index) {
          return Container(
            margin: const EdgeInsets.only(right: 8),
            child: ActionChip(
              label: Text(
                replies[index],
                style: const TextStyle(
                  color: AppTheme.primaryColor,
                  fontSize: 14,
                ),
              ),
              backgroundColor: Colors.white,
              side: const BorderSide(color: AppTheme.primaryColor),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(20),
              ),
              onPressed: () => onReplyTap(replies[index]),
            ),
          );
        },
      ),
    );
  }
}