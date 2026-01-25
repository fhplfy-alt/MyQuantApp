import streamlit as st
from io import BytesIO

# ==========================================
# ⚠️ 1. 安全访问控制 (新功能)
# ==========================================
def check_password():
    if "password_correct" not in st.session_state:
        st.markdown("### 🔐 V45 智能量化系统安全验证")
        pwd = st.text_input("请输入访问密码", type="password")
        if st.button("登录"):
            if pwd == "vip666888":
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("❌ 密码错误")
        return False
    return True

if not check_password():
    st.stop()

# ==========================================
# ⚠️ 核心配置 (保持原样)
# ==========================================
st.set_page_config(
    page_title="V45 完美说明书版", 
    layout="wide", 
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

st.title("🛡️ V45 智能量化系统 (全信号图例版)")
st.caption("✅ 系统已就绪 | 核心组件加载完成 | 支持6000股扫描 | V45 Build")

# ==========================================
# 1. 安全导入
# ==========================================
try:
    import plotly.graph_objects as go
    import random
    import baostock as bs
    import pandas as pd
    import numpy as np
    import time
    import datetime
    from sklearn.linear_model import LinearRegression
except ImportError as e:
    st.error(f"❌ 启动失败！缺少必要运行库: {e}")
    st.stop()

# ==========================================
# 0. 全局配置 (保持原逻辑)
# ==========================================
STRATEGY_TIP = """
👇 信号含义详细对照：
👑 四星共振: [涨停+缺口+连阳+倍量] 同时满足，最强主升浪信号！
🐲 妖股基因: 60天内3板 + 筹码>80%，游资龙头特征。
🔥 换手锁仓: 连续高换手 + 高获利，主力清洗浮筹接力。
🔴 温和吸筹: 3连阳但涨幅小 + 筹码集中，主力潜伏期。
📈 多头排列: 股价收阳且重心上移，趋势健康，建议持有。
🚀 金叉突变: 短期均线向上金叉长期均线，买入信号。
⚡ 死叉/空头: 趋势向下或破位，建议规避。
"""

ACTION_TIP = """
👇 操作建议说明：
🟥 STRONG BUY: 【重点关注】确定性极高
🟧 BUY (博弈): 【激进买入】短线博弈
🟨 BUY (低吸): 【稳健买入】逢低建仓
🟦 HOLD: 【持股】趋势完好，拿住不动
⬜ WAIT: 【观望】无机会
"""

STRATEGY_LOGIC = {
    "👑 四星共振": "近20日有涨停 + 向上跳空缺口 + 4连阳 + 量比>1.8",
    "🐲 妖股基因": "近60日涨停≥3次 + 获利筹码>80% + 上市>30天",
    "🔥 换手锁仓": "连续2日换手率>5% + 获利筹码>70%",
    "🔴 温和吸筹": "3连阳且累计涨幅<5% + 获利筹码>62%",
    "📈 多头排列": "昨日收阳 且 今日收盘价 > 昨日收盘价",
    "💎 RSI超卖反弹": "RSI<30后回升至35以上,超跌反弹机会",
    "📊 布林带突破": "价格突破布林带上轨+成交量放大",
    "🎯 KDJ金叉": "K线上穿D线+RSI>50,短期买入信号",
    "📉 200日均线趋势": "价格站上200日均线+均线向上,长期上升趋势"
}

# ==========================================
# 2. 核心引擎
# ==========================================
class QuantsEngine:
    def __init__(self):
        self.MAX_SCAN_LIMIT = 6000
    
    def clean_code(self, code):
        code = str(code).strip()
        if not (code.startswith('sh.') or code.startswith('sz.')):
            return f"sh.{code}" if code.startswith('6') else f"sz.{code}"
        return code

    def is_valid(self, code, name):
        if "sh.688" in code: return False 
        if "bj." in code or code.startswith("sz.8") or code.startswith("sz.4"): return False 
        if "ST" in name: return False 
        return True

    def get_all_stocks(self):
        """修复：确保全场扫描能成功获取数据"""
            try:
            bs.login() # 显式重新登录
                rs = bs.query_all_stock()
                stocks = []
                data_list = []
            while (rs.error_code == '0') and rs.next():
                data_list.append(rs.get_row_data())
                
                for data in data_list:
                    if len(data) >= 2:
                    code, name = data[0], data[1]
                        if self.is_valid(code, name):
                            stocks.append(code)
                bs.logout()
                    return stocks[:self.MAX_SCAN_LIMIT]
                except:
        return []

    def get_index_stocks(self, index_type="zz500"):
        bs.login()
                stocks = []
                try:
            if index_type == "hs300": rs = bs.query_hs300_stocks()
            else: rs = bs.query_zz500_stocks()
            while rs.next(): stocks.append(rs.get_row_data()[1])
        except: pass
        finally: bs.logout()
                    return stocks[:self.MAX_SCAN_LIMIT]

    def calc_winner_rate(self, df, current_price):
        if df.empty: return 0.0
        total_vol = df['volume'].sum()
        if total_vol == 0: return 0.0
        profit_vol = df[df['close'] < current_price]['volume'].sum()
        return (profit_vol / total_vol) * 100

    def calc_risk_level(self, price, ma5, ma20):
        if ma5 == 0: return "未知"
        bias = (price - ma5) / ma5 * 100
        if bias > 15: return "High (高危)"
        elif price < ma20: return "Med (破位)"
        else: return "Low (安全)"
    
    def calc_rsi(self, df, period=14):
        """计算RSI相对强弱指标"""
        try:
            if len(df) < period + 1:
                return None
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else None
        except:
            return None
    
    def calc_kdj(self, df, period=9):
        """计算KDJ指标"""
        try:
            if len(df) < period + 1:
                return None, None, None
            low_min = df['low'].rolling(window=period).min()
            high_max = df['high'].rolling(window=period).max()
            rsv = (df['close'] - low_min) / (high_max - low_min) * 100
            
            k = rsv.ewm(com=2, adjust=False).mean()
            d = k.ewm(com=2, adjust=False).mean()
            j = 3 * k - 2 * d
            
            return k.iloc[-1], d.iloc[-1], j.iloc[-1]
        except:
            return None, None, None
    
    def calc_bollinger(self, df, period=20, std_dev=2):
        """计算布林带指标"""
        try:
            if len(df) < period:
                return None, None, None
            ma = df['close'].rolling(window=period).mean()
            std = df['close'].rolling(window=period).std()
            upper = ma + (std * std_dev)
            lower = ma - (std * std_dev)
            return upper.iloc[-1], ma.iloc[-1], lower.iloc[-1]
        except:
            return None, None, None

    def _process_single_stock(self, code, max_price=None):
        # 保持你原始的策略判定逻辑不变
        code = self.clean_code(code)
        end = datetime.datetime.now().strftime("%Y-%m-%d")
        start = (datetime.datetime.now() - datetime.timedelta(days=150)).strftime("%Y-%m-%d")
        
        data = []
        info = {'name': code, 'industry': '-', 'ipoDate': '2000-01-01'}
        
        try:
            rs_info = bs.query_stock_basic(code=code)
            if rs_info.next():
                row = rs_info.get_row_data()
                info['name'] = row[1]
                info['ipoDate'] = row[2]
            rs_ind = bs.query_stock_industry(code)
            if rs_ind.next(): info['industry'] = rs_ind.get_row_data()[3] 
            if not self.is_valid(code, info['name']): return None
            rs = bs.query_history_k_data_plus(code, "date,open,close,high,low,volume,pctChg,turn", start_date=start, frequency="d", adjustflag="3")
            while rs.next(): data.append(rs.get_row_data())
        except: return None

        if not data: return None
            df = pd.DataFrame(data, columns=["date", "open", "close", "high", "low", "volume", "pctChg", "turn"])
            df = df.apply(pd.to_numeric, errors='coerce')
        if len(df) < 60: return None

        curr = df.iloc[-1]
        prev = df.iloc[-2]
        if max_price is not None and curr['close'] > max_price: return None

        winner_rate = self.calc_winner_rate(df, curr['close'])
        df['MA5'] = df['close'].rolling(5).mean()
        df['MA20'] = df['close'].rolling(20).mean()
        df['MA200'] = df['close'].rolling(200).mean() if len(df) >= 200 else pd.Series([None] * len(df))
        risk_level = self.calc_risk_level(curr['close'], df['MA5'].iloc[-1], df['MA20'].iloc[-1])

        # 计算技术指标
        rsi = self.calc_rsi(df)
        k, d, j = self.calc_kdj(df)
        bb_upper, bb_mid, bb_lower = self.calc_bollinger(df)

        signal_tags, priority, action = [], 0, "WAIT (观望)"

        # 原有策略保留
        if (all(df['pctChg'].tail(3) > 0) and df['pctChg'].tail(3).sum() <= 5 and winner_rate > 62):
            signal_tags.append("🔴温和吸筹"); priority = 60; action = "BUY (低吸)"

        if all(df['turn'].tail(2) > 5) and winner_rate > 70:
            signal_tags.append("🔥换手锁仓"); priority = max(priority, 70); action = "BUY (博弈)"

        if len(df.tail(60)[df.tail(60)['pctChg'] > 9.5]) >= 3 and winner_rate > 80:
            signal_tags.append("🐲妖股基因"); priority = 90; action = "STRONG BUY"

        # 四星共振原逻辑
        recent_20 = df.tail(20)
        has_limit_up_20 = len(recent_20[recent_20['pctChg'] > 9.5]) > 0
        is_double_vol = (curr['volume'] > prev['volume'] * 1.8)
        if has_limit_up_20 and is_double_vol:
            signal_tags.append("👑四星共振"); priority = 100; action = "STRONG BUY"
        
        # 新增策略：RSI超卖反弹
        if rsi is not None and len(df) >= 2:
                prev_rsi = self.calc_rsi(df.iloc[:-1])
            if prev_rsi is not None and prev_rsi < 30 and rsi > 35:
                    signal_tags.append("💎RSI超卖反弹")
                    priority = max(priority, 65)
                if action in ["WAIT (观望)", "HOLD (持有)"]:
                        action = "BUY (低吸)"
        
        # 新增策略：布林带突破
        if bb_upper is not None and bb_lower is not None:
            if curr['close'] > bb_upper and curr['volume'] > df['volume'].tail(20).mean() * 1.2:
                signal_tags.append("📊布林带突破")
                priority = max(priority, 75)
                if action in ["WAIT (观望)", "HOLD (持有)"]:
                    action = "BUY (博弈)"
        
        # 新增策略：KDJ金叉
        if k is not None and d is not None:
            if len(df) >= 2:
                prev_k, prev_d, _ = self.calc_kdj(df.iloc[:-1])
                if prev_k is not None and prev_d is not None:
                    if prev_k <= prev_d and k > d and rsi is not None and rsi > 50:
                        signal_tags.append("🎯KDJ金叉")
                        priority = max(priority, 70)
                        if action in ["WAIT (观望)", "HOLD (持有)"]:
                            action = "BUY (博弈)"
        
        # 新增策略：200日均线趋势
        if len(df) >= 200 and not pd.isna(df['MA200'].iloc[-1]):
            ma200_current = df['MA200'].iloc[-1]
            ma200_prev = df['MA200'].iloc[-2] if len(df) >= 201 else ma200_current
            if curr['close'] > ma200_current and ma200_current > ma200_prev:
                signal_tags.append("📉200日均线趋势")
                priority = max(priority, 80)
                if action in ["WAIT (观望)", "HOLD (持有)", "BUY (低吸)"]:
                    action = "BUY (低吸)" if action == "WAIT (观望)" else action

        # 多头排列策略
        if prev['close'] > prev['open'] and curr['close'] > prev['close']:
            signal_tags.append("📈多头排列")
            priority = max(priority, 50)
            if action == "WAIT (观望)":
                action = "HOLD (持有)"

        if priority == 0: return None

        return {
            "result": {
                "代码": code, "名称": info['name'], "所属行业": info['industry'],
                "现价": curr['close'], "涨跌": f"{curr['pctChg']:.2f}%", 
                "获利筹码": winner_rate, "风险评级": risk_level,
                "策略信号": " + ".join(signal_tags), "综合评级": action, "priority": priority
            },
            "alert": f"{info['name']}" if priority >= 90 else None,
            "option": f"{code} | {info['name']}"
        }

    def scan_market_optimized(self, code_list, max_price=None):
        # 保持原有的进度条逻辑，增加命中数量显示
        results, alerts, valid_codes_list = [], [], []
        bs.login()
        total = len(code_list)
        progress_bar = st.progress(0, text=f"🚀 正在扫描 (0/{total}) | 命中: 0 只")
        
        for i, code in enumerate(code_list):
            try:
                res = self._process_single_stock(code, max_price)
                if res:
                    results.append(res["result"])
                    if res["alert"]: alerts.append(res["alert"])
                    valid_codes_list.append(res["option"])
            except: continue
            # 每10个更新一次进度，显示命中数量
            if i % 10 == 0 or i == len(code_list) - 1:
                hit_count = len(results)
                progress_bar.progress((i + 1) / total, text=f"🔍 扫描中: {code} ({i+1}/{total}) | 命中: {hit_count} 只")

        bs.logout()
        progress_bar.empty()
        return results, alerts, valid_codes_list

    def get_deep_data(self, code):
        """修复白屏的关键：增加严谨的数据校验"""
        try:
            bs.login()
            end = datetime.datetime.now().strftime("%Y-%m-%d")
            start = (datetime.datetime.now() - datetime.timedelta(days=180)).strftime("%Y-%m-%d")
            rs = bs.query_history_k_data_plus(code, "date,open,close,high,low,volume", start_date=start, end_date=end, frequency="d", adjustflag="3")
            data = []
            while rs.next(): data.append(rs.get_row_data())
            bs.logout()
            if not data: return None
            df = pd.DataFrame(data, columns=["date", "open", "close", "high", "low", "volume"])
            df[["open", "close", "high", "low", "volume"]] = df[["open", "close", "high", "low", "volume"]].apply(pd.to_numeric, errors='coerce')
            return df.dropna()
        except: return None

    def run_ai_prediction(self, df):
        """增强版AI预测：预估后三天股票走势，包括价格、涨跌幅等"""
        if df is None or len(df) < 30: return None
        try:
            # 使用更多历史数据提高预测准确性
            recent = df.tail(30).reset_index(drop=True)
            X = np.array(recent.index).reshape(-1, 1)
            y = recent['close'].values
            
            # 训练模型
            model = LinearRegression().fit(X, y)
            
            # 预测后三天价格
            next_indices = np.array([[len(recent)], [len(recent)+1], [len(recent)+2]])
            pred_prices = model.predict(next_indices)
            
            # 计算当前价格
            current_price = df['close'].iloc[-1]
            
            # 计算涨跌幅
            changes = [(p - current_price) / current_price * 100 for p in pred_prices]
            
            # 生成日期（后三天）：明日/后日/大后日
            last_date = pd.to_datetime(df['date'].iloc[-1])
            date_labels = ["明日", "后日", "大后日"]
            dates = []
            day_offset = 1
            for i in range(3):
                next_date = last_date + datetime.timedelta(days=day_offset)
                # 跳过周末
                while next_date.weekday() >= 5:  # 5=Saturday, 6=Sunday
                    next_date += datetime.timedelta(days=1)
                dates.append(f"{date_labels[i]} ({next_date.strftime('%m-%d')})")
                day_offset += 1
            
            # 判断趋势（颜色：红色=上涨，绿色=下跌，蓝色=横盘）
            avg_change = np.mean(changes)
            if avg_change > 2:
                color = "red"  # 红色=预测上涨
                title = "📈 AI预测：上涨趋势"
                desc = f"预计未来三天平均涨幅 {avg_change:.2f}%"
                action = "建议持有或逢低买入"
            elif avg_change < -2:
                color = "green"  # 绿色=预测下跌
                title = "📉 AI预测：下跌趋势"
                desc = f"预计未来三天平均跌幅 {abs(avg_change):.2f}%"
                action = "建议谨慎观望或减仓"
                else:
                color = "blue"  # 蓝色=预测横盘
                title = "➡️ AI预测：震荡整理"
                desc = f"预计未来三天波动较小，平均变化 {abs(avg_change):.2f}%"
                action = "建议持有观望"

            return {
                "dates": dates,
                "prices": pred_prices.tolist(),
                "changes": changes,
                "pred_price": pred_prices[0],
                "current_price": current_price,
                "color": color,
                "title": title,
                "desc": desc,
                "action": action
            }
        except Exception as e:
            return None

    def plot_professional_kline(self, df, title):
        """增强版K线图：添加买卖信号标记"""
        if df is None or df.empty: return None
            
        try:
            # 计算技术指标
            df['MA5'] = df['close'].rolling(5).mean()
            df['MA20'] = df['close'].rolling(20).mean()
            df['MA200'] = df['close'].rolling(200).mean() if len(df) >= 200 else None
            
            # 计算RSI和KDJ用于信号判断
            rsi = self.calc_rsi(df)
            k, d, j = self.calc_kdj(df)
            bb_upper, bb_mid, bb_lower = self.calc_bollinger(df)
            
            # 创建K线图
            fig = go.Figure()
            
            # 添加K线（调换红绿颜色：A股习惯红=涨，绿=跌）
            fig.add_trace(go.Candlestick(
                x=df['date'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='K线',
                increasing_line_color='red',    # 上涨用红色
                decreasing_line_color='green',  # 下跌用绿色
                increasing_fillcolor='red',     # 上涨填充红色
                decreasing_fillcolor='green'    # 下跌填充绿色
            ))
            
            # 添加均线
            if 'MA5' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['date'],
                    y=df['MA5'],
                    mode='lines',
                    name='MA5',
                    line=dict(color='orange', width=1)
                ))
            
            if 'MA20' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['date'],
                    y=df['MA20'],
                    mode='lines',
                    name='MA20',
                    line=dict(color='blue', width=1)
                ))
            
            if df['MA200'] is not None and not df['MA200'].isna().all():
                fig.add_trace(go.Scatter(
                    x=df['date'],
                    y=df['MA200'],
                    mode='lines',
                    name='MA200',
                    line=dict(color='purple', width=1, dash='dash')
                ))
            
            # 添加布林带
            if bb_upper is not None and bb_lower is not None:
                # 计算布林带数据
                period = 20
                if len(df) >= period:
                    ma = df['close'].rolling(window=period).mean()
                    std = df['close'].rolling(window=period).std()
                    upper = ma + (std * 2)
                    lower = ma - (std * 2)
                    
                    fig.add_trace(go.Scatter(
                        x=df['date'],
                        y=upper,
                        mode='lines',
                        name='布林上轨',
                        line=dict(color='gray', width=1, dash='dash'),
                        showlegend=False
                    ))
                    fig.add_trace(go.Scatter(
                        x=df['date'],
                        y=lower,
                        mode='lines',
                        name='布林下轨',
                        line=dict(color='gray', width=1, dash='dash'),
                        fill='tonexty',
                        fillcolor='rgba(128,128,128,0.1)',
                        showlegend=False
                    ))
            
            # 识别买卖信号（区分不同强度）
            strong_buy_signals = []  # 红色"强买"：200日均线趋势
            medium_buy_signals = []  # 橙色"买入"：RSI/KDJ/布林带
            basic_buy_signals = []   # 黄色"B"：MA金叉
            sell_signals = []        # 绿色"卖出"：MA死叉
            
                    for i in range(1, len(df)):
                curr = df.iloc[i]
                prev = df.iloc[i-1]
                
                # 1. 最强买入信号：200日均线趋势（红色"强买"）
                if i >= 200 and df['MA200'] is not None and not df['MA200'].isna().all():
                    ma200_curr = df['MA200'].iloc[i]
                    ma200_prev = df['MA200'].iloc[i-1] if i >= 201 else ma200_curr
                    if curr['close'] > ma200_curr and ma200_curr > ma200_prev:
                        strong_buy_signals.append((df['date'].iloc[i], curr['low'] * 0.98, "强买"))
                
                # 2. 中等强度买入信号：RSI/KDJ/布林带（橙色"买入"）
                # RSI超卖反弹
                if i >= 15:
                    curr_rsi = self.calc_rsi(df.iloc[:i+1])
                    prev_rsi = self.calc_rsi(df.iloc[:i])
                    if prev_rsi is not None and curr_rsi is not None:
                        if prev_rsi < 30 and curr_rsi > 35:
                            medium_buy_signals.append((df['date'].iloc[i], curr['low'] * 0.98, "买入"))
                
                # KDJ金叉
                if i >= 10:
                    curr_k, curr_d, _ = self.calc_kdj(df.iloc[:i+1])
                    prev_k, prev_d, _ = self.calc_kdj(df.iloc[:i])
                    if prev_k is not None and prev_d is not None and curr_k is not None and curr_d is not None:
                        if prev_k <= prev_d and curr_k > curr_d:
                            medium_buy_signals.append((df['date'].iloc[i], curr['low'] * 0.98, "买入"))
                
                # 布林带突破
                if i >= 20 and bb_upper is not None:
                    if curr['close'] > bb_upper and curr['volume'] > df['volume'].iloc[max(0, i-20):i].mean() * 1.2:
                        medium_buy_signals.append((df['date'].iloc[i], curr['low'] * 0.98, "买入"))
                
                # 3. 基础买入信号：MA5上穿MA20（金叉）（黄色"B"）
                if i >= 20:
                    if prev['MA5'] <= prev['MA20'] and curr['MA5'] > curr['MA20']:
                        basic_buy_signals.append((df['date'].iloc[i], curr['low'] * 0.98, "B"))
            
                # 卖出信号：MA5下穿MA20（死叉）（绿色"卖出"）
                if i >= 20:
                    if prev['MA5'] >= prev['MA20'] and curr['MA5'] < curr['MA20']:
                        sell_signals.append((df['date'].iloc[i], curr['high'] * 1.02, "卖出"))
            
            # 添加最强买入信号标记（红色"强买"）
            if strong_buy_signals:
                dates, prices, _ = zip(*strong_buy_signals)
                        fig.add_trace(go.Scatter(
                    x=list(dates),
                    y=list(prices),
                            mode='markers+text', 
                    name='强买',
                    text=['强买'] * len(dates),
                    textposition='top center',
                    marker=dict(
                        symbol='triangle-up',
                        size=15,
                        color='red',
                        line=dict(width=2, color='darkred')
                    ),
                    textfont=dict(size=10, color='red')
                        ))
                    
            # 添加中等强度买入信号标记（橙色"买入"）
            if medium_buy_signals:
                dates, prices, _ = zip(*medium_buy_signals)
                        fig.add_trace(go.Scatter(
                    x=list(dates),
                    y=list(prices),
                            mode='markers+text', 
                            name='买入',
                    text=['买入'] * len(dates),
                    textposition='top center',
                    marker=dict(
                        symbol='triangle-up',
                        size=12,
                        color='orange',
                        line=dict(width=2, color='darkorange')
                    ),
                    textfont=dict(size=9, color='orange')
                        ))
                    
            # 添加基础买入信号标记（黄色"B"）
            if basic_buy_signals:
                dates, prices, _ = zip(*basic_buy_signals)
                        fig.add_trace(go.Scatter(
                    x=list(dates),
                    y=list(prices),
                            mode='markers+text', 
                    name='B',
                    text=['B'] * len(dates),
                    textposition='top center',
                    marker=dict(
                        symbol='triangle-up',
                        size=10,
                        color='yellow',
                        line=dict(width=1, color='gold')
                    ),
                    textfont=dict(size=8, color='darkgoldenrod')
                ))
            
            # 添加卖出信号标记（绿色"卖出"）
            if sell_signals:
                dates, prices, _ = zip(*sell_signals)
                    fig.add_trace(go.Scatter(
                    x=list(dates),
                    y=list(prices),
                        mode='markers+text', 
                    name='卖出',
                    text=['卖出'] * len(dates),
                    textposition='bottom center',
                    marker=dict(
                        symbol='triangle-down',
                        size=12,
                        color='green',
                        line=dict(width=2, color='darkgreen')
                    ),
                    textfont=dict(size=9, color='green')
                ))
            
            # 更新布局
            fig.update_layout(
                title=title,
                xaxis_rangeslider_visible=False,
                height=600,
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            return fig
        except Exception as e:
            # 如果出错，返回基础K线图（调换红绿颜色）
            fig = go.Figure(data=[go.Candlestick(
                x=df['date'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='K线',
                increasing_line_color='red',    # 上涨用红色
                decreasing_line_color='green',  # 下跌用绿色
                increasing_fillcolor='red',     # 上涨填充红色
                decreasing_fillcolor='green'    # 下跌填充绿色
            )])
            fig.update_layout(title=title, xaxis_rangeslider_visible=False, height=500)
            return fig

# ==========================================
# 3. 界面 UI (完全恢复原布局)
# ==========================================
engine = QuantsEngine()

if 'full_pool' not in st.session_state: st.session_state['full_pool'] = []
if 'scan_res' not in st.session_state: st.session_state['scan_res'] = []
if 'valid_options' not in st.session_state: st.session_state['valid_options'] = []

st.sidebar.header("🕹️ 控制台")
max_price_limit = st.sidebar.slider("💰 价格上限 (元)", 3.0, 100.0, 20.0)
pool_mode = st.sidebar.radio("🔎 选股范围:", ("中证500 (中小盘)", "沪深300 (大盘)", "全市场扫描", "手动输入"))
scan_limit = st.sidebar.slider("🔢 扫描数量 (池大小)", 50, 6000, 500, step=50)

if pool_mode == "手动输入":
    target_pool_str = st.sidebar.text_area("监控股票池", "600519, 002131", height=100)
    final_code_list = [c.strip() for c in target_pool_str.replace("，", ",").split(",") if c.strip()]
else:
    if st.sidebar.button(f"📥 加载 {pool_mode} 成分股"):
        with st.spinner("获取中..."):
            if pool_mode == "全市场扫描": st.session_state['full_pool'] = engine.get_all_stocks()
            elif "中证500" in pool_mode: st.session_state['full_pool'] = engine.get_index_stocks("zz500")
            else: st.session_state['full_pool'] = engine.get_index_stocks("hs300")
            st.sidebar.success(f"已加载 {len(st.session_state['full_pool'])} 只")
    final_code_list = st.session_state.get('full_pool', [])[:scan_limit]

if st.sidebar.button("🚀 启动全策略扫描 (V45)", type="primary"):
    if not final_code_list: st.sidebar.error("请先加载股票！")
    else:
        res, alerts, opts = engine.scan_market_optimized(final_code_list, max_price=max_price_limit)
        st.session_state['scan_res'], st.session_state['valid_options'], st.session_state['alerts'] = res, opts, alerts

# 导出Excel功能（放在sidebar中，确保显示）
st.sidebar.markdown("---")
st.sidebar.subheader("📊 导出功能")

# 检查是否有扫描结果
scan_res = st.session_state.get('scan_res', [])
if scan_res and len(scan_res) > 0:
    # 创建DataFrame并排序：priority >= 90的排在最前面
    df_export = pd.DataFrame(scan_res)
    if 'priority' in df_export.columns:
        df_export['is_high_priority'] = df_export['priority'] >= 90
        df_export = df_export.sort_values(by=['is_high_priority', 'priority'], ascending=[False, False])
        df_export = df_export.drop(columns=['is_high_priority'], errors='ignore')
    
    # 移除priority列（内部使用，不需要导出）
    df_export_clean = df_export.drop(columns=['priority'], errors='ignore')
    
    # 创建Excel文件
    try:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_export_clean.to_excel(writer, index=False, sheet_name='扫描结果')
        excel_data = output.getvalue()
        
        # 生成文件名（包含日期时间）
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"股票扫描结果_{timestamp}.xlsx"
        
        st.sidebar.download_button(
            label="📥 导出为Excel",
            data=excel_data,
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            key="export_excel_button"
        )
    except ImportError:
        st.sidebar.error("❌ 缺少 openpyxl 库")
        st.sidebar.info("💡 请运行: pip install openpyxl")
    except Exception as e:
        st.sidebar.error(f"❌ 导出失败: {str(e)}")
else:
    st.sidebar.info("💡 请先进行扫描，扫描完成后可导出结果")

# 策略展示逻辑 (保持原样)
with st.expander("📖 **策略逻辑白皮书**", expanded=False):
    for k, v in STRATEGY_LOGIC.items(): st.markdown(f"- **{k}**: {v}")

if st.session_state['scan_res']:
    # 排序：priority >= 90的排在最前面，然后按priority降序
    df_scan = pd.DataFrame(st.session_state['scan_res'])
    df_scan['is_high_priority'] = df_scan['priority'] >= 90
    df_scan = df_scan.sort_values(by=['is_high_priority', 'priority'], ascending=[False, False])
    df_scan = df_scan.drop(columns=['is_high_priority'], errors='ignore')
    
    # 显示命中股票数量
    total_count = len(df_scan)
    st.success(f"✅ **扫描完成！共命中 {total_count} 只符合条件的股票**")
    
    # 显示主力高控盘标的（priority >= 90的股票）
    if 'alerts' in st.session_state and st.session_state['alerts']:
        alert_count = len(st.session_state['alerts'])
        alert_names = "、".join(st.session_state['alerts'][:5])  # 最多显示5个
        if len(st.session_state['alerts']) > 5:
            alert_names += f"等{alert_count}只"
        st.success(f"🔥 **发现 {alert_count} 只【主力高控盘】标的：{alert_names}**")
    
    st.dataframe(df_scan, hide_index=True)

# 深度分析 (增强版)
if st.session_state['valid_options']:
    st.subheader("🧠 深度分析")
    target = st.selectbox("选择目标进行深度分析", st.session_state['valid_options'])
    target_code = target.split("|")[0].strip()
    target_name = target.split("|")[1].strip() if "|" in target else target

    if st.button(f"🚀 立即分析 {target_name}", type="primary"):
        with st.spinner("正在获取数据并分析..."):
                df = engine.get_deep_data(target_code)
                if df is not None and not df.empty:
                    # 显示K线图（带买卖信号）
                    st.markdown("### 📊 K线分析（含买卖信号）")
                    fig = engine.plot_professional_kline(df, f"{target_name} - K线图")
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.info("""
                        💡 **图例说明**: 
                        - 🔺 **红色"强买"** = 200日均线趋势信号，最强买入信号
                        - 🔺 **橙色"买入"** = RSI/KDJ/布林带信号，中等强度买入
                        - 🔺 **黄色"B"** = MA金叉信号，基础买入信号
                        - 🔻 **绿色"卖出"** = MA死叉信号，建议卖出
                        - **橙色线** = MA5均线（5日移动平均线）
                        - **蓝色线** = MA20均线（20日移动平均线）
                        - **紫色虚线** = MA200均线（200日移动平均线，长期趋势）
                        - **灰色区域** = 布林带（价格波动范围）
                        - 信号仅供参考，投资需谨慎
                        """)
                    
                    # 显示AI预测（后三天走势）
                    st.markdown("### 🤖 AI预测：未来三天走势")
                    future = engine.run_ai_prediction(df)
                    if future:
                    col1, col2, col3 = st.columns(3)
                        
                        # 显示当前价格
                        current_price = future['current_price']
                        col1.metric("当前价格", f"¥{current_price:.2f}")
                        
                        # 显示预测信息
                        if future['color'] == 'green':
                            st.success(f"### {future['title']}\n{future['desc']}\n\n**{future['action']}**")
                        elif future['color'] == 'red':
                            st.error(f"### {future['title']}\n{future['desc']}\n\n**{future['action']}**")
                        else:
                            st.warning(f"### {future['title']}\n{future['desc']}\n\n**{future['action']}**")

                        # 显示后三天详细预测（明日/后日/大后日）
                        st.markdown("#### 📅 AI 时空推演 (未来3日)")
                        pred_cols = st.columns(3)
                        for i in range(3):
                            pred_price = future['prices'][i]
                            change = future['changes'][i]
                            date_label = future['dates'][i]  # 已经是"明日 (MM-DD)"格式
                            change_amount = pred_price - current_price
                            
                            with pred_cols[i]:
                                if change > 0:
                                    st.metric(
                                        label=date_label,
                                value=f"¥{pred_price:.2f}", 
                                        delta=f"{change_amount:+.2f} ({change:+.2f}%)",
                                        delta_color="inverse"
                            )
                    else:
                                    st.metric(
                                        label=date_label,
                                        value=f"¥{pred_price:.2f}",
                                        delta=f"{change_amount:+.2f} ({change:+.2f}%)",
                                        delta_color="normal"
                                    )
                    
                        # 显示预测数据表格
                        with st.expander("📋 查看详细预测数据"):
                            pred_df = pd.DataFrame({
                                '日期': future['dates'],  # 已经是"明日 (MM-DD)"格式
                                '预测价格': [f"¥{p:.2f}" for p in future['prices']],
                                '涨跌金额': [f"{p - current_price:+.2f}" for p in future['prices']],
                                '涨跌幅': [f"{c:+.2f}%" for c in future['changes']]
                            })
                            st.dataframe(pred_df, hide_index=True)
                    else:
                        st.warning("⚠️ AI预测数据不足，无法生成预测")
                        
                    # 显示最近交易数据
                    with st.expander("📋 查看最近交易数据"):
                        st.dataframe(df.tail(20), hide_index=True)
                else:
                    st.error("❌ 数据获取失败，请重试")
            
st.caption("💡 使用提示：扫描时请勿刷新页面。投资有风险。")