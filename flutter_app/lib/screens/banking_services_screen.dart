import 'package:flutter/material.dart';
import '../widgets/banking_info_card.dart';

class BankingServicesScreen extends StatelessWidget {
  const BankingServicesScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Dịch vụ HDBank'),
      ),
      body: ListView(
        padding: const EdgeInsets.symmetric(vertical: 16),
        children: [
          BankingInfoCard(
            title: 'Thẻ tín dụng',
            description: 'Thông tin về các loại thẻ tín dụng, hạn mức, lãi suất',
            icon: Icons.credit_card,
            onTap: () => _showServiceInfo(context, 'Thẻ tín dụng'),
          ),
          BankingInfoCard(
            title: 'Thẻ ghi nợ',
            description: 'Thẻ ATM, thẻ thanh toán, rút tiền',
            icon: Icons.payment,
            onTap: () => _showServiceInfo(context, 'Thẻ ghi nợ'),
          ),
          BankingInfoCard(
            title: 'Tài khoản thanh toán',
            description: 'Mở tài khoản, quản lý số dư, lịch sử giao dịch',
            icon: Icons.account_balance,
            onTap: () => _showServiceInfo(context, 'Tài khoản thanh toán'),
          ),
          BankingInfoCard(
            title: 'Vay vốn',
            description: 'Vay tiêu dùng, vay mua nhà, vay kinh doanh',
            icon: Icons.monetization_on,
            onTap: () => _showServiceInfo(context, 'Vay vốn'),
          ),
          BankingInfoCard(
            title: 'Tiết kiệm',
            description: 'Gửi tiết kiệm, lãi suất, kỳ hạn',
            icon: Icons.savings,
            onTap: () => _showServiceInfo(context, 'Tiết kiệm'),
          ),
          BankingInfoCard(
            title: 'Chuyển tiền',
            description: 'Chuyển khoản trong nước, quốc tế, phí giao dịch',
            icon: Icons.send,
            onTap: () => _showServiceInfo(context, 'Chuyển tiền'),
          ),
        ],
      ),
    );
  }

  void _showServiceInfo(BuildContext context, String service) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text(service),
        content: Text('Thông tin chi tiết về $service sẽ được cập nhật sớm.'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Đóng'),
          ),
        ],
      ),
    );
  }
}