# 决定：Kestrel 的失败告警只发到 #kestrel-oncall

邮件告警容易被其他通知淹没，也没法统计确认时间，所以我们不再发邮件。现在，Kestrel
任务连续失败三次后，告警只会出现在 Slack 的 #kestrel-oncall 频道里；值班同学收到后要在
10 分钟内确认。
