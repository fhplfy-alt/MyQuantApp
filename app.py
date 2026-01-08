import streamlit as st
import time
import datetime

# ==========================================
# ⚠️ 核心配置
# ==========================================
st.set_page_config(
    page_title="V70 智能诊断版", 
    layout="wide", 
    page_icon="🩺",
    initial_sidebar_state="expanded"
)

st.title("🩺 V70 智能量化系统 (全功能·自动查错)")
st.caption("✅ 实时监控运行状态 | ✅ 自动报告错误原因")

# ==========================================
# 1. 安全导入
# ==========================================
try:
    import plotly.graph_objects as go
    import baostock as bs
    import pandas as pd
    import numpy as np
    import urllib.request
    import json
    from sklearn.linear_model import LinearRegression
    import threading
    import pdfplumber
except ImportError as e:
    st.error(f"❌ 缺少库: {e}")
    st.stop()

# ==========================================
# 0. 全局配置
# ==========================================
bs_lock = threading.Lock()

STRATEGY_TIP = """
🌤️ 首阳首板: 涨停后缩量回调，今日再收阳 (N字反转)
🤐 极度缩量: 量能萎缩至5日均量一半 (洗盘特征)
👑 四星共振: [涨停+缺口+连阳+倍量] 最强主升
🐲 妖股基因: 60天内3板 + 筹码>80%
🔥 换手锁仓: 高换手 + 高获利
"""

ACTION_TIP = """
🟥 STRONG BUY: 【重仓】四星共振/首阳首板
🟧 BUY (博弈): 【激进】换手锁仓/接力
🟨 BUY (低吸): 【潜伏】温和吸筹/缩量回踩
🟦 HOLD: 【持股】趋势完好
⬜ WAIT: 【观望】无机会
"""

STRATEGY_LOGIC = {
    "🌤️ 首阳首板": "涨停后回调2-8天 + 不破支撑 + 今日收阳",
    "🤐 极度缩量": "今日成交量 < 5日均量 * 0.6",
    "👑 四星共振": "近20日有涨停 + 向上跳空缺口 + 4连阳 + 量比>1.8",
    "🐲 妖股基因": "近60日涨停≥3次 + 获利筹码>80% + 上市>30天",
    "🔥 换手锁仓": "连续2日换手率>5% + 获利筹码>70%",
    "🔴 温和吸筹": "3连阳且累计涨幅<5% + 获利筹码>62%",
    "📈 多头排列": "昨日收阳 且 今日收盘价 > 昨日收盘价"
}

ALL_INDUSTRIES = [
    "农林牧渔", "采掘", "化工", "钢铁", "有色金属", "电子", "家用电器", "食品饮料", 
    "纺织服装", "轻工制造", "医药生物", "公用事业", "交通运输", "房地产", "商业贸易", 
    "休闲服务", "综合", "建筑材料", "建筑装饰", "电气设备", "国防军工", "计算机", 
    "传媒", "通信", "银行", "非银金融", "汽车", "机械设备"
]

# ==========================================
# 2. 核心引擎
# ==========================================
class QuantsEngine:
    def __init__(self):
        pass

    def clean_code(self, code):
        code = str(code).strip()
        if not (code.startswith('sh.') or code.startswith('sz.')):
            if code.startswith('6'): return f"sh.{code}"
            elif code.startswith('8') or code.startswith('4'): return f"bj.{code}"
            else: return f"sz.{code}"
        return code

    def get_market_sentiment(self):
        bs.login()
        try:
            end = datetime.datetime.now().strftime("%Y-%m-%d")
            start = (datetime.datetime.now() - datetime.timedelta(days=100)).strftime("%Y-%m-%d")
            rs = bs.query_history_k_data_plus("sh.000001", "date,close", start_date=start, end_date=end, frequency="d", adjustflag="3")
            data = []
            while rs.next(): data.append(rs.get_row_data())
            if not data: return None
            df = pd.DataFrame(data, columns=["date", "close"])
            df['close'] = df['close'].astype(float)
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            dif = exp1 - exp2
            dea = dif.ewm(span=9, adjust=False).mean()
            if dif.iloc[-1] > dea.iloc[-1]:
                return {"status": "强市 (金叉)", "color": "red", "pos": "80%"}
            else:
                return {"status": "弱市 (死叉)", "color": "green", "pos": "0-20%"}
        except: return None
        finally: bs.logout()

    def get_realtime_quote(self, code):
        try:
            clean_code = code.split('.')[-1]
            market_id = "1" if code.startswith("sh") else "0"
            if code.startswith("bj"): return None
            url = f"https://push2.eastmoney.com/api/qt/stock/get?invt=2&fltt=2&fields=f43,f44,f45,f46,f47,f48,f60,f168,f170&secid={market_id}.{clean_code}"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=2) as f:
                d = json.loads(f.read().decode('utf-8')).get('data')
                if d:
                    cp = float(d['f43'])
                    if cp == 0: cp = float(d['f60'])
                    return {'date': datetime.date.today().strftime("%Y-%m-%d"), 'open': float(d['f46']), 'pre_close': float(d['f60']), 'close': cp, 'high': float(d['f44']), 'low': float(d['f45']), 'volume': float(d['f47'])*100, 'turn': float(d['f168'])}
        except: return None
        return None

    def get_all_stocks(self):
        bs.login()
        stocks = []
        try:
            for i in range(5):
                date = (datetime.datetime.now() - datetime.timedelta(days=i)).strftime("%Y-%m-%d")
                rs = bs.query_all_stock(day=date)
                temp = []
                while rs.next():
                    if rs.get_row_data()[1] == '1': temp.append(rs.get_row_data()[0])
                if len(temp) > 1000:
                    stocks = temp; break
        except: pass
        finally: bs.logout()
        
        # 如果获取不到，使用指数保底
        if len(stocks) < 100:
             return self.get_index_stocks("hs300") + self.get_index_stocks("zz500")
        return stocks

    def get_index_stocks(self, index_type="zz500"):
        bs.login()
        stocks = []
        try:
            rs = bs.query_zz500_stocks() if index_type == "zz500" else bs.query_hs300_stocks()
            while rs.next(): stocks.append(rs.get_row_data()[1])
        except: pass
        finally: bs.logout()
        return stocks

    def is_valid(self, code, name, industry, allow_kc, allow_bj, selected_industries):
        if "ST" in name: return False
        if "sh.688" in code and not allow_kc: return False
        if ("bj." in code or code.startswith("sz.8")) and not allow_bj: return False
        if selected_industries:
            is_match = False
            for ind in selected_industries:
                if ind in str(industry):
                    is_match = True; break
            if not is_match: return False
        return True

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

    def _process_single_stock(self, code, max_price, allow_kc, allow_bj, selected_industries):
        code = self.clean_code(code)
        end = datetime.datetime.now().strftime("%Y-%m-%d")
        start = (datetime.datetime.now() - datetime.timedelta(days=150)).strftime("%Y-%m-%d")
        
        data = []
        info = {'name': code, 'industry': '未分类', 'ipoDate': '2000-01-01'}
        
        try:
            rs_info = bs.query_stock_basic(code=code)
            if rs_info.error_code != '0': return None 
            if rs_info.next():
                row = rs_info.get_row_data()
                info['name'] = row[1]
                info['ipoDate'] = row[2]
            
            rs_ind = bs.query_stock_industry(code)
            if rs_ind.next():
                info['industry'] = rs_ind.get_row_data()[3] 

            if not self.is_valid(code, info['name'], info['industry'], allow_kc, allow_bj, selected_industries): return None

            rs = bs.query_history_k_data_plus(code, "date,open,close,high,low,volume,pctChg,turn", start_date=start, frequency="d", adjustflag="3")
            while rs.next(): data.append(rs.get_row_data())
        except:
            return None

        if not data: return None
        try:
            df = pd.DataFrame(data, columns=["date", "open", "close", "high", "low", "volume", "pctChg", "turn"]).apply(pd.to_numeric, errors='coerce')
        except: return None
        if len(df) < 60: return None

        rt = self.get_realtime_quote(code)
        if rt and rt['close'] > 0:
            if str(df.iloc[-1]['date']) != str(rt['date']):
                pct = (rt['close'] - rt['pre_close']) / rt['pre_close'] * 100
                new = pd.DataFrame([{"date": rt['date'], "open": rt['open'], "close": rt['close'], "high": rt['high'], "low": rt['low'], "volume": rt['volume'], "pctChg": pct, "turn": rt['turn']}])
                df = pd.concat([df, new], ignore_index=True)
            else:
                idx = df.index[-1]
                df.at[idx, 'close'] = rt['close']; df.at[idx, 'high'] = rt['high']; df.at[idx, 'low'] = rt['low']; df.at[idx, 'volume'] = rt['volume']
                df.at[idx, 'pctChg'] = (rt['close'] - rt['pre_close']) / rt['pre_close'] * 100

        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        if max_price is not None:
            if curr['close'] > max_price: return None

        winner_rate = self.calc_winner_rate(df, curr['close'])
        
        try: ipo_date = datetime.datetime.strptime(info['ipoDate'], "%Y-%m-%d")
        except: ipo_date = datetime.datetime(2000, 1, 1)
        days_listed = (datetime.datetime.now() - ipo_date).days

        df['MA5'] = df['close'].rolling(5).mean()
        df['MA10'] = df['close'].rolling(10).mean()
        df['MA20'] = df['close'].rolling(20).mean()
        risk_level = self.calc_risk_level(curr['close'], df['MA5'].iloc[-1], df['MA20'].iloc[-1])

        signal_tags = []
        priority = 0
        action = "WAIT"

        # 策略集合
        recent_10 = df.tail(10).iloc[:-1]
        has_limit_recent = len(recent_10[recent_10['pctChg'] > 9.5]) > 0
        is_today_red = curr['close'] > curr['open']
        is_correction = prev['close'] < df.tail(5)['high'].max()
        if has_limit_recent and is_today_red and is_correction:
            signal_tags.append("🌤️首阳首板"); priority = 95; action = "STRONG BUY"

        vol_ma5 = df['volume'].tail(6).iloc[:-1].mean()
        if curr['volume'] < vol_ma5 * 0.6: 
            signal_tags.append("🤐极度缩量"); priority = max(priority, 5)

        if all(df['pctChg'].tail(3) > 0) and df['pctChg'].tail(3).sum() <= 5 and winner_rate > 62:
            signal_tags.append("🔴温和吸筹"); priority = max(priority, 60); action = "BUY (低吸)"
        
        turn_val = df['turn'].iloc[-1] if df['turn'].iloc[-1] > 0 else df['turn'].iloc[-2]
        prev_turn = df['turn'].iloc[-2]
        if (turn_val > 5 and prev_turn > 5) and winner_rate > 70:
            signal_tags.append("🔥换手锁仓"); priority = max(priority, 70); action = "BUY (博弈)"
            
        limit_60 = len(df.tail(60)[df.tail(60)['pctChg'] > 9.5])
        if limit_60 >= 3 and winner_rate > 80:
            signal_tags.append("🐲妖股基因"); priority = max(priority, 90); action = "STRONG BUY"

        has_limit_20 = len(df.tail(20)[df.tail(20)['pctChg'] > 9.5]) > 0
        is_double = curr['volume'] > prev['volume'] * 1.8
        is_red4 = (df['close'].tail(4) > df['open'].tail(4)).all()
        if has_limit_20 and is_red4 and is_double:
            signal_tags.append("👑四星共振"); priority = 100; action = "STRONG BUY"
            
        elif prev['open'] < prev['close'] and curr['close'] > prev['close']:
             if priority == 0:
                 signal_tags.append("📈多头排列"); priority = 10; action = "HOLD"

        if priority == 0: return None

        return {
            "result": {
                "代码": code, "名称": info['name'], "行业": info['industry'], 
                "现价": curr['close'], "涨跌": f"{curr['pctChg']:.2f}%", 
                "获利筹码": winner_rate, "风险评级": risk_level,
                "策略信号": " + ".join(signal_tags),
                "综合评级": action, "priority": priority
            },
            "alert": f"{info['name']}" if priority >= 90 else None,
            "option": f"{code} | {info['name']}"
        }

    # 🔥🔥🔥 V70 核心：带诊断功能的扫描 🔥🔥🔥
    def scan_market(self, code_list, max_price, allow_kc, allow_bj, selected_industries):
        results, alerts, codes = [], [], []
        
        # 1. 登录检查
        lg = bs.login()
        if lg.error_code != '0':
            st.error(f"❌ Baostock 登录失败 (Error {lg.error_code})，请检查网络！")
            return [], [], []

        progress_bar = st.progress(0, text=f"🚀 正在扫描 {len(code_list)} 只股票...")
        status_text = st.empty() # 状态显示框
        total = len(code_list)
        
        fail_count = 0
        
        # 2. 循环处理
        for i, c in enumerate(code_list):
            if i % 2 == 0:
                progress_bar.progress((i + 1) / total, text=f"🔍 正在分析: {c} ({i+1}/{total})")
                status_text.text(f"📊 已命中: {len(results)} 只 | 失败: {fail_count} 只")
            try:
                time.sleep(0.01)
                r = self._process_single_stock(c, max_price, allow_kc, allow_bj, selected_industries)
                if r: 
                    results.append(r["result"])
                    if r["alert"]: alerts.append(r["alert"])
                    codes.append(r["option"])
            except: 
                fail_count += 1
                # 尝试重连
                bs.logout(); time.sleep(0.5); bs.login()
                continue

        bs.logout()
        progress_bar.empty()
        status_text.empty()
        
        # 🔥 如果扫描完了还是 0，给出详细建议
        if len(results) == 0:
            st.warning(f"""
            ⚠️ 扫描完成，但没有找到符合条件的股票。
            可能原因：
            1. **价格过滤太严**：当前上限 {max_price} 元，建议调高到 50 或 100 元。
            2. **行业过滤太严**：建议清空行业选择（全选）。
            3. **网络问题**：有 {fail_count} 只股票获取失败。
            """)
            
        return results, alerts, codes

    @st.cache_data(ttl=600)
    def get_deep(_self, code):
        bs.login()
        try:
            end = datetime.datetime.now().strftime("%Y-%m-%d")
            start = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
            rs = bs.query_history_k_data_plus(code, "date,open,close,high,low,volume,peTTM,pctChg", start_date=start, end_date=end, frequency="d", adjustflag="3")
            data = [r for r in rs.get_data()]
            bs.logout()
            if not data: return None
            return pd.DataFrame(data, columns=["date", "open", "close", "high", "low", "volume", "peTTM", "pctChg"]).apply(pd.to_numeric, errors='coerce').dropna()
        except:
            bs.logout()
            return None

    def run_ai_prediction(self, df):
        if len(df) < 30: return None
        recent = df.tail(30).reset_index(drop=True)
        X = np.array(recent.index).reshape(-1, 1)
        y = recent['close'].values
        model = LinearRegression()
        model.fit(X, y)
        last_idx = recent.index[-1]
        future_idx = np.array([[last_idx + 1], [last_idx + 2], [last_idx + 3]])
        pred_prices = model.predict(future_idx)
        
        future_dates = []
        current_date = datetime.date.today()
        for i in range(1, 4):
            d = current_date + datetime.timedelta(days=i)
            future_dates.append(d.strftime("%Y-%m-%d"))

        slope = model.coef_[0]
        last_price = df['close'].iloc[-1]
        
        if slope > 0.05:
            hint_title = "🚀 上升通道加速中"
            hint_desc = f"惯性推演：股价将在 **{future_dates[1]}** 尝试冲击 **¥{pred_prices[1]:.2f}**。"
            action = "建议：坚定持有 / 逢低买入"
            color = "red"
        elif slope > 0:
            hint_title = "📈 震荡缓慢上行"
            hint_desc = f"趋势温和，预计 **{future_dates[1]}** 到达 **¥{pred_prices[1]:.2f}**。"
            action = "建议：耐心持股"
            color = "red"
        elif slope < -0.05:
            hint_title = "📉 下跌趋势加速"
            hint_desc = f"空头较强，预计 **{future_dates[1]}** 回落至 **¥{pred_prices[1]:.2f}**。"
            action = "建议：反弹卖出"
            color = "green"
        else:
            hint_title = "⚖️ 横盘震荡"
            hint_desc = f"多空平衡，预计 **{future_dates[1]}** 在 **¥{pred_prices[1]:.2f}** 震荡。"
            action = "建议：观望"
            color = "blue"

        return {
            "dates": future_dates,
            "prices": pred_prices,
            "pred_price": pred_prices[0],
            "title": hint_title,
            "desc": hint_desc,
            "action": action,
            "color": color
        }

    def calc_indicators(self, df):
        df = df.copy()
        df['MA5'] = df['close'].rolling(5).mean()
        df['MA20'] = df['close'].rolling(20).mean()
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['DIF'] = exp1 - exp2
        df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
        df['MACD'] = 2 * (df['DIF'] - df['DEA'])
        return df

    def plot_professional_kline(self, df, title):
        df['Signal'] = 0
        df.loc[(df['MA5'] > df['MA20']) & (df['MA5'].shift(1) <= df['MA20'].shift(1)), 'Signal'] = 1 
        df.loc[(df['MA5'] < df['MA20']) & (df['MA5'].shift(1) >= df['MA20'].shift(1)), 'Signal'] = -1 

        buy_points = df[df['Signal'] == 1]
        sell_points = df[df['Signal'] == -1]

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
            name='K线', increasing_line_color='red', decreasing_line_color='green'
        ))
        fig.add_trace(go.Scatter(x=df['date'], y=df['MA5'], name='MA5', line=dict(color='orange', width=1)))
        fig.add_trace(go.Scatter(x=df['date'], y=df['MA20'], name='MA20', line=dict(color='blue', width=1)))

        if not buy_points.empty:
            fig.add_trace(go.Scatter(x=buy_points['date'], y=buy_points['low']*0.98, mode='markers+text', marker=dict(symbol='triangle-up', size=12, color='red'), text='B', textposition='bottom center', name='买入'))
        
        if not sell_points.empty:
            fig.add_trace(go.Scatter(x=sell_points['date'], y=sell_points['high']*1.02, mode='markers+text', marker=dict(symbol='triangle-down', size=12, color='green'), text='S', textposition='top center', name='卖出'))

        fig.update_layout(title=f"{title} - 智能操盘K线 (含B/S点)", xaxis_rangeslider_visible=False, height=600)
        return fig

# ==========================================
# 3. 界面 UI
# ==========================================
engine = QuantsEngine()

st.sidebar.header("🕹️ 战神控制台")
max_price_limit = st.sidebar.slider("💰 价格上限 (元)", 3.0, 100.0, 20.0)

st.sidebar.markdown("#### 🏭 行业过滤")
selected_industries = st.sidebar.multiselect("行业 (留空全选):", options=ALL_INDUSTRIES, default=[])
allow_kc = st.sidebar.checkbox("包含科创板 (688)", value=False)
allow_bj = st.sidebar.checkbox("包含北交所 (8xx)", value=False)

mode = st.sidebar.radio("选股范围", ("全市场精选", "手动输入"))
limit = st.sidebar.slider("🔢 扫描数量", 100, 6000, 200)

if mode == "手动输入":
    default_pool = "600519, 002131, 002312, 600580, 002594"
    target_pool_str = st.sidebar.text_area("监控股票池", default_pool, height=100)
    final_code_list = target_pool_str.replace("，", ",").split(",")
else:
    if st.sidebar.button("📥 加载全市场"):
        with st.spinner("正在遍历交易所数据库..."):
            st.session_state['pool'] = engine.get_all_stocks()
            st.sidebar.success(f"已加载全量 {len(st.session_state['pool'])} 只")
    
    if 'pool' in st.session_state:
        pool_len = len(st.session_state['pool'])
        st.sidebar.info(f"市场总数: {pool_len} | 本次扫描前 {limit} 只")
    
    pool = st.session_state.get('pool', [])[:limit]

if st.sidebar.button("🚀 启动战神扫描"):
    # 调用新的扫描函数
    res, al, opts = engine.scan_market_optimized(pool, max_price_limit, allow_kc, allow_bj, selected_industries)
    
    st.session_state['res'] = res
    st.session_state['valid_options'] = opts
    st.session_state['alerts'] = al

if st.session_state.get('al'): 
    names = "、".join(st.session_state['al'])
    st.success(f"🔥 发现 {len(st.session_state['al'])} 只【主力高控盘】标的：**{names}**")

with st.expander("📖 **策略逻辑白皮书 (透明度报告)**", expanded=False):
    st.markdown("##### 🔍 核心策略定义")
    for k, v in STRATEGY_LOGIC.items(): st.markdown(f"- **{k}**: {v}")

if st.session_state.get('res'):
    st.dataframe(pd.DataFrame(st.session_state['res']), use_container_width=True, 
                 column_config={
                     "获利筹码": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
                     "风险评级": st.column_config.TextColumn(help="基于乖离率计算"),
                     "策略信号": st.column_config.TextColumn(help=STRATEGY_TIP, width="large"),
                     "综合评级": st.column_config.TextColumn(help=ACTION_TIP, width="medium")
                 })

st.divider()

if st.session_state.get('valid_options'):
    st.subheader("🧠 深度分析")
    target = st.selectbox("选择目标", st.session_state['valid_options'])
    
    target_code = target.split("|")[0].strip()
    target_name = target.split("|")[1].strip()

    if st.button(f"🚀 分析 {target_name}"):
        with st.spinner("AI 正在深度运算..."):
            
            df = engine.get_deep(target_code)
            rt = engine.get_realtime_quote(target_code)
            
            if df is not None:
                if rt:
                    if str(df.iloc[-1]['date']) != str(rt['date']):
                         new = pd.DataFrame([{"date":rt['date'], "open":rt['open'], "close":rt['close'], "high":rt['high'], "low":rt['low'], "volume":rt['volume'], "peTTM":0, "pctChg": 0}])
                         df = pd.concat([df, new], ignore_index=True)
                
                df['MA5'] = df['close'].rolling(5).mean(); df['MA10'] = df['close'].rolling(10).mean()
                future_info = engine.run_ai_prediction(df)
                
                last_limit_idx = df[df['pctChg'] > 9.5].last_valid_index()
                if last_limit_idx:
                    limit_row = df.loc[last_limit_idx]
                    support_half = (limit_row['open'] + limit_row['close']) / 2
                    wash_days = len(df) - 1 - last_limit_idx
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("当前价格", f"¥{df.iloc[-1]['close']:.2f}")
                    c2.metric("🛡️ 首板1/2强支撑", f"¥{support_half:.2f}", help="跌破此位需止损")
                    c3.metric("🔵 10日生命线", f"¥{df.iloc[-1]['MA10']:.2f}")
                    c4.metric("🚿 洗盘天数", f"{wash_days}天")
                else:
                    st.info("近期无涨停")

                if future_info:
                    st.markdown("---")
                    if future_info['color'] == 'red':
                        st.error(f"### {future_info['title']}\n{future_info['desc']}")
                    else:
                        st.info(f"### {future_info['title']}\n{future_info['desc']}")

                fig = engine.plot_professional_kline(df, target.split("|")[1])
                st.plotly_chart(fig, use_container_width=True)
                st.success("✅ **战法解析**：请重点关注 **蓝色10日线** 与 **1/2支撑位**。")

# 研报
st.sidebar.markdown("---")
if st.sidebar.checkbox("📄 启用研报分析"):
    st.subheader("📄 智能文档分析器")
    uploaded_file = st.file_uploader("上传 PDF 研报/财报", type="pdf")
    if uploaded_file and st.button("开始分析"):
        with pdfplumber.open(uploaded_file) as pdf:
            text = "".join([p.extract_text() for p in pdf.pages[:5]])
            st.success("分析完成！")
            c1, c2 = st.columns(2)
            c1.info("🔥 **利好关键词**")
            for w in ["增长", "新高", "龙头", "受益"]: 
                if w in text: c1.write(f"✅ {w}")
            c2.warning("⚠️ **风险关键词**")
            for w in ["下降", "亏损", "风险", "减持"]: 
                if w in text: c2.write(f"❌ {w}")
            st.text_area("文档摘要预览", text[:1000], height=300)