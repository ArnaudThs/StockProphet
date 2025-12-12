from multiticker_refactor.pipeline_multi import build_multi_ticker_dataset

print('Testing sentiment integration...')
df, metadata = build_multi_ticker_dataset(
    tickers=['AAPL'],
    start_date='2020-01-01',
    end_date='2020-02-28',
    include_rnn=False,
    include_sentiment=True,
    verbose=True
)

print(f'\n✅ Success! Dataset shape: {df.shape}')
print(f'Sentiment column exists: {"AAPL_Sentiment" in df.columns}')
if 'AAPL_Sentiment' in df.columns:
    non_zero = (df['AAPL_Sentiment'] != 0).sum()
    print(f'Non-zero sentiment days: {non_zero}/{len(df)}')
