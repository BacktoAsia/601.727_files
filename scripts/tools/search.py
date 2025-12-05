"""
Finance Search Agent using yfinance - Specialized agent for retrieving financial 
information from Yahoo Finance using the yfinance library.
"""

import os
import json
import re
from typing import Optional, List, Dict, Tuple, Union
from datetime import datetime, timedelta
import warnings

# Suppress yfinance warnings
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("Warning: yfinance not installed. Install it with: pip install yfinance")


class FinanceSearchAgent:
    """
    A specialized search agent for retrieving financial information using yfinance.
    Provides access to stock prices, company information, financial statements,
    market data, and historical data from Yahoo Finance.
    """
    
    def __init__(
        self,
        cache_enabled: bool = True,
        cache: Optional[Dict] = None,
        timeout: int = 10
    ):
        """
        Initialize the Finance Search Agent.
        
        Args:
            cache_enabled: Whether to cache results to avoid repeated API calls
            cache: Optional cache dictionary (shared across instances)
            timeout: Request timeout in seconds
        """
        if not YFINANCE_AVAILABLE:
            raise ImportError(
                "yfinance is not installed. Please install it with: pip install yfinance"
            )
        
        self.cache_enabled = cache_enabled
        self.cache = cache or {}
        self.timeout = timeout
        
    def _get_cache_key(self, ticker: str, data_type: str) -> str:
        """Generate cache key for a ticker and data type."""
        return f"{ticker}_{data_type}"
    
    def _get_ticker(self, symbol: str) -> yf.Ticker:
        """
        Get yfinance Ticker object for a symbol.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
            
        Returns:
            yf.Ticker object
        """
        return yf.Ticker(symbol.upper())
    
    def search_stock(
        self, 
        symbol: str, 
        include_history: bool = True,
        period: str = "1mo"
    ) -> Dict:
        """
        Search for stock information by ticker symbol.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
            include_history: Whether to include historical price data
            period: Period for historical data (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            
        Returns:
            Dictionary containing stock information
        """
        symbol = symbol.upper().strip()
        cache_key = self._get_cache_key(symbol, 'stock_info')
        
        # Check cache
        if self.cache_enabled and cache_key in self.cache:
            print(f"Using cached data for {symbol}")
            return self.cache[cache_key]
        
        try:
            ticker = self._get_ticker(symbol)
            info = ticker.info
            
            result = {
                'symbol': symbol,
                'company_name': info.get('longName') or info.get('shortName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap'),
                'current_price': info.get('currentPrice') or info.get('regularMarketPrice'),
                'previous_close': info.get('previousClose'),
                'open': info.get('open'),
                'day_low': info.get('dayLow'),
                'day_high': info.get('dayHigh'),
                '52_week_low': info.get('fiftyTwoWeekLow'),
                '52_week_high': info.get('fiftyTwoWeekHigh'),
                'volume': info.get('volume'),
                'average_volume': info.get('averageVolume'),
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'peg_ratio': info.get('pegRatio'),
                'dividend_yield': info.get('dividendYield'),
                'dividend_rate': info.get('dividendRate'),
                'beta': info.get('beta'),
                'eps': info.get('trailingEps'),
                'book_value': info.get('bookValue'),
                'price_to_book': info.get('priceToBook'),
                'revenue': info.get('totalRevenue'),
                'profit_margin': info.get('profitMargins'),
                'operating_margin': info.get('operatingMargins'),
                'ebitda': info.get('ebitda'),
                'debt_to_equity': info.get('debtToEquity'),
                'return_on_equity': info.get('returnOnEquity'),
                'return_on_assets': info.get('returnOnAssets'),
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange', 'N/A'),
                'website': info.get('website', 'N/A'),
                'description': info.get('longBusinessSummary', 'N/A'),
                'employees': info.get('fullTimeEmployees'),
                'timestamp': datetime.now().isoformat()
            }
            
            # Add historical data if requested
            if include_history:
                try:
                    hist = ticker.history(period=period)
                    if not hist.empty:
                        result['historical_data'] = {
                            'period': period,
                            'start_date': hist.index[0].strftime('%Y-%m-%d'),
                            'end_date': hist.index[-1].strftime('%Y-%m-%d'),
                            'latest_close': float(hist['Close'].iloc[-1]),
                            'latest_volume': int(hist['Volume'].iloc[-1]),
                            'data_points': len(hist),
                            'summary': {
                                'avg_close': float(hist['Close'].mean()),
                                'min_close': float(hist['Close'].min()),
                                'max_close': float(hist['Close'].max()),
                                'avg_volume': float(hist['Volume'].mean())
                            }
                        }
                except Exception as e:
                    result['historical_data'] = {'error': str(e)}
            
            # Cache the result
            if self.cache_enabled:
                self.cache[cache_key] = result
            
            return result
            
        except Exception as e:
            return {
                'symbol': symbol,
                'error': f"Failed to retrieve data for {symbol}: {str(e)}",
                'timestamp': datetime.now().isoformat()
            }
    
    def search_multiple_stocks(
        self, 
        symbols: List[str],
        include_history: bool = False
    ) -> Dict[str, Dict]:
        """
        Search for multiple stocks at once.
        
        Args:
            symbols: List of stock ticker symbols
            include_history: Whether to include historical data
            
        Returns:
            Dictionary mapping symbols to their data
        """
        results = {}
        for symbol in symbols:
            results[symbol] = self.search_stock(symbol, include_history=include_history)
        return results
    
    def get_historical_prices(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d"
    ) -> Dict:
        """
        Get historical price data for a stock.
        
        Args:
            symbol: Stock ticker symbol
            period: Period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            Dictionary containing historical price data
        """
        symbol = symbol.upper().strip()
        cache_key = self._get_cache_key(symbol, f'history_{period}_{interval}')
        
        if self.cache_enabled and cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            ticker = self._get_ticker(symbol)
            hist = ticker.history(period=period, interval=interval)
            
            if hist.empty:
                return {
                    'symbol': symbol,
                    'error': 'No historical data available',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Convert to list of dictionaries for JSON serialization
            hist_data = []
            for date, row in hist.iterrows():
                hist_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': int(row['Volume']),
                    'adj_close': float(row['Adj Close']) if 'Adj Close' in row else None
                })
            
            result = {
                'symbol': symbol,
                'period': period,
                'interval': interval,
                'data_points': len(hist_data),
                'start_date': hist.index[0].strftime('%Y-%m-%d'),
                'end_date': hist.index[-1].strftime('%Y-%m-%d'),
                'data': hist_data,
                'summary': {
                    'first_close': float(hist['Close'].iloc[0]),
                    'last_close': float(hist['Close'].iloc[-1]),
                    'min_close': float(hist['Close'].min()),
                    'max_close': float(hist['Close'].max()),
                    'avg_close': float(hist['Close'].mean()),
                    'total_volume': int(hist['Volume'].sum())
                },
                'timestamp': datetime.now().isoformat()
            }
            
            if self.cache_enabled:
                self.cache[cache_key] = result
            
            return result
            
        except Exception as e:
            return {
                'symbol': symbol,
                'error': f"Failed to retrieve historical data: {str(e)}",
                'timestamp': datetime.now().isoformat()
            }
    
    def get_financial_statements(
        self,
        symbol: str
    ) -> Dict:
        """
        Get financial statements (income statement, balance sheet, cash flow).
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary containing financial statements
        """
        symbol = symbol.upper().strip()
        cache_key = self._get_cache_key(symbol, 'financials')
        
        if self.cache_enabled and cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            ticker = self._get_ticker(symbol)
            
            result = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat()
            }
            
            # Income statement
            try:
                financials = ticker.financials
                if not financials.empty:
                    result['income_statement'] = financials.to_dict('index')
            except Exception as e:
                result['income_statement'] = {'error': str(e)}
            
            # Balance sheet
            try:
                balance_sheet = ticker.balance_sheet
                if not balance_sheet.empty:
                    result['balance_sheet'] = balance_sheet.to_dict('index')
            except Exception as e:
                result['balance_sheet'] = {'error': str(e)}
            
            # Cash flow
            try:
                cashflow = ticker.cashflow
                if not cashflow.empty:
                    result['cashflow'] = cashflow.to_dict('index')
            except Exception as e:
                result['cashflow'] = {'error': str(e)}
            
            if self.cache_enabled:
                self.cache[cache_key] = result
            
            return result
            
        except Exception as e:
            return {
                'symbol': symbol,
                'error': f"Failed to retrieve financial statements: {str(e)}",
                'timestamp': datetime.now().isoformat()
            }
    
    def get_market_data(
        self,
        symbols: List[str]
    ) -> Dict:
        """
        Get current market data for multiple symbols.
        
        Args:
            symbols: List of stock ticker symbols
            
        Returns:
            Dictionary containing market data
        """
        try:
            # Use yfinance's download for multiple symbols
            data = yf.download(
                ' '.join([s.upper() for s in symbols]),
                period="1d",
                interval="1m",
                progress=False,
                group_by='ticker'
            )
            
            result = {
                'symbols': symbols,
                'timestamp': datetime.now().isoformat(),
                'data': {}
            }
            
            for symbol in symbols:
                symbol_upper = symbol.upper()
                if symbol_upper in data.columns.levels[0] if hasattr(data.columns, 'levels') else symbol_upper in data.columns:
                    ticker_data = data[symbol_upper] if hasattr(data.columns, 'levels') else data
                    if not ticker_data.empty:
                        latest = ticker_data.iloc[-1]
                        result['data'][symbol_upper] = {
                            'close': float(latest['Close']),
                            'volume': int(latest['Volume']),
                            'high': float(latest['High']),
                            'low': float(latest['Low']),
                            'open': float(latest['Open'])
                        }
            
            return result
            
        except Exception as e:
            return {
                'symbols': symbols,
                'error': f"Failed to retrieve market data: {str(e)}",
                'timestamp': datetime.now().isoformat()
            }
    
    def search_company(self, symbol: str) -> Dict:
        """
        Search for comprehensive company information.
        Alias for search_stock with full details.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary containing company information
        """
        return self.search_stock(symbol, include_history=True)
    
    def format_results(self, data: Dict, format_type: str = 'json') -> str:
        """
        Format search results for display.
        
        Args:
            data: Data dictionary from search methods
            format_type: Format type ('json', 'markdown', 'text')
            
        Returns:
            Formatted string
        """
        if format_type == 'json':
            return json.dumps(data, indent=2, ensure_ascii=False)
        
        elif format_type == 'markdown':
            if 'error' in data:
                return f"# Error\n\n{data['error']}\n"
            
            md = f"# {data.get('company_name', data.get('symbol', 'Finance Data'))}\n\n"
            
            if 'symbol' in data:
                md += f"**Symbol:** {data['symbol']}\n\n"
            
            # Key metrics section
            if 'current_price' in data:
                md += "## Key Metrics\n\n"
                metrics = [
                    ('Current Price', data.get('current_price')),
                    ('Market Cap', self._format_number(data.get('market_cap'))),
                    ('PE Ratio', data.get('pe_ratio')),
                    ('Dividend Yield', self._format_percent(data.get('dividend_yield'))),
                    ('52 Week High', data.get('52_week_high')),
                    ('52 Week Low', data.get('52_week_low')),
                    ('Volume', self._format_number(data.get('volume'))),
                    ('Beta', data.get('beta')),
                ]
                
                for label, value in metrics:
                    if value is not None:
                        md += f"- **{label}:** {value}\n"
                md += "\n"
            
            # Company info
            if 'sector' in data:
                md += "## Company Information\n\n"
                md += f"- **Sector:** {data.get('sector', 'N/A')}\n"
                md += f"- **Industry:** {data.get('industry', 'N/A')}\n"
                md += f"- **Exchange:** {data.get('exchange', 'N/A')}\n"
                if data.get('website') and data['website'] != 'N/A':
                    md += f"- **Website:** {data['website']}\n"
                md += "\n"
            
            # Description
            if 'description' in data and data['description'] != 'N/A':
                md += f"## Description\n\n{data['description'][:500]}...\n\n"
            
            # Historical data summary
            if 'historical_data' in data and 'error' not in data['historical_data']:
                hist = data['historical_data']
                md += "## Historical Data Summary\n\n"
                md += f"- **Period:** {hist.get('period', 'N/A')}\n"
                md += f"- **Latest Close:** ${hist.get('latest_close', 'N/A'):.2f}\n"
                md += f"- **Average Close:** ${hist.get('summary', {}).get('avg_close', 'N/A'):.2f}\n"
                md += f"- **Min Close:** ${hist.get('summary', {}).get('min_close', 'N/A'):.2f}\n"
                md += f"- **Max Close:** ${hist.get('summary', {}).get('max_close', 'N/A'):.2f}\n\n"
            
            return md
        
        else:  # text format
            if 'error' in data:
                return f"Error: {data['error']}\n"
            
            text = f"{data.get('company_name', data.get('symbol', 'Finance Data'))}\n"
            text += f"Symbol: {data.get('symbol', 'N/A')}\n\n"
            
            if 'current_price' in data:
                text += f"Current Price: ${data.get('current_price', 'N/A')}\n"
                text += f"Market Cap: {self._format_number(data.get('market_cap'))}\n"
                text += f"PE Ratio: {data.get('pe_ratio', 'N/A')}\n"
                text += f"Dividend Yield: {self._format_percent(data.get('dividend_yield'))}\n"
            
            return text
    
    def _format_number(self, num: Optional[float]) -> str:
        """Format large numbers with K, M, B suffixes."""
        if num is None:
            return 'N/A'
        if num >= 1e12:
            return f"${num/1e12:.2f}T"
        elif num >= 1e9:
            return f"${num/1e9:.2f}B"
        elif num >= 1e6:
            return f"${num/1e6:.2f}M"
        elif num >= 1e3:
            return f"${num/1e3:.2f}K"
        else:
            return f"${num:.2f}"
    
    def _format_percent(self, num: Optional[float]) -> str:
        """Format number as percentage."""
        if num is None:
            return 'N/A'
        return f"{num * 100:.2f}%"


def create_finance_search_agent(**kwargs) -> FinanceSearchAgent:
    """
    Factory function to create a FinanceSearchAgent instance.
    
    Args:
        **kwargs: Arguments to pass to FinanceSearchAgent
        
    Returns:
        FinanceSearchAgent instance
    """
    return FinanceSearchAgent(**kwargs)


# Example usage
if __name__ == "__main__":
    if not YFINANCE_AVAILABLE:
        print("=" * 60)
        print("Finance Search Agent - Missing yfinance Library")
        print("=" * 60)
        print("\nTo use this script, you need to install the yfinance library:")
        print("  pip install yfinance")
        print("\nThis module can be imported without yfinance installed.")
        print("The library is only required when performing actual searches.")
        print("=" * 60)
        exit(0)
    
    # Create finance search agent
    agent = create_finance_search_agent(cache_enabled=True)
    
    # Example searches
    print("=" * 60)
    print("Example 1: Search for stock information (AAPL)")
    print("=" * 60)
    results = agent.search_stock("AAPL", include_history=True, period="3mo")
    print(agent.format_results(results, format_type='markdown'))
    
    print("\n" + "=" * 60)
    print("Example 2: Search for company information (MSFT)")
    print("=" * 60)
    results = agent.search_company("MSFT")
    print(agent.format_results(results, format_type='markdown'))
    
    print("\n" + "=" * 60)
    print("Example 3: Get historical prices (TSLA)")
    print("=" * 60)
    results = agent.get_historical_prices("TSLA", period="6mo")
    print(f"Symbol: {results['symbol']}")
    print(f"Period: {results['period']}")
    print(f"Data Points: {results['data_points']}")
    print(f"Summary: {json.dumps(results['summary'], indent=2)}")
    
    print("\n" + "=" * 60)
    print("Example 4: Search multiple stocks")
    print("=" * 60)
    results = agent.search_multiple_stocks(["GOOGL", "AMZN", "META"], include_history=False)
    for symbol, data in results.items():
        if 'error' not in data:
            print(f"\n{symbol}: {data.get('company_name')} - ${data.get('current_price', 'N/A')}")

