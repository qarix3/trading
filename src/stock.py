total_shares = 350
stock_price = 25.75
total = total_shares * stock_price

brokerage = total * 0.006
total_brokerage = 70 + brokerage

total_cost_investment = total + total_brokerage

print("Initial Investment:RM " + str(total))
print("% Brokerage :RM " + str(brokerage))
print("Total Brokerage Fee :RM " + str(total_brokerage))
print("Cost of Investment:RM " + str(total_cost_investment))

net_sale = 350 * 30.00
sales_commission = (net_sale * 0.007) + 50
profit = net_sale - sales_commission
rate_of_return= net_sale - profit

print("Total profit:RM " + str(profit))
print("Rate of Return:RM " + str(rate_of_return))