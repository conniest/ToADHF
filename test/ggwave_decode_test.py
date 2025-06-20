import ggwave

ctx = ggwave.init()

test = "CQ CQ DE KN6UBF"
encoded = ggwave.encode(test, protocolId=1, volume=100)
result = ggwave.decode(ctx, encoded)
print(result.decode() if result else "Nothing decoded")
