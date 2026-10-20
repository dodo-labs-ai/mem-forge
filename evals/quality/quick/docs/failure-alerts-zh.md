# 决定：Kestrel 的失败告警只发到 #kestrel-oncall

Kestrel 任务连续失败三次后，告警只发送到 Slack 的 #kestrel-oncall 频道，不再发邮件。
值班同学需要在 15 分钟内确认告警。我们放弃了邮件告警，因为邮件容易被其他通知淹没，
也无法统计确认时间。
