import streamlit as st
import time
import datetime

# ==========================================
# ⚠️ 核心配置
# ==========================================
st.set_page_config(
    page_title="V65 交互修复版", 
    layout="wide", 
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

st.title("🛡️ V65 智能量化系统 (最终稳定内核)")
st.caption("✅ 已修复点击无响应问题 | ✅ 深度分析3次重试")

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
        if len(stocks) < 100:
             return self.get_index_stocks_backup()
        return stocks

    def get_index_stocks_backup(self):
        bs.login()
        stocks = []
        try:
            rs = bs.query_zz500_stocks()
            while rs.next(): stocks.append(rs.get_row_data()[1])
            rs2 = bs.query_hs300_stocks()
            while rs2.next(): stocks.append(rs2.get_row_data()[1])
            stocks = list(set(stocks))
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
            df = pd.DataFrame(data, columns=["date", "open", "close", "high", "low", "volume", "pctChg", "turn"])
            df = df.apply(pd.to_numeric, errors='coerce')
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
                df.at[idx, 'close'] = rt['close']
                df.at[idx, 'high'] = rt['high']
                df.at[idx, 'low'] = rt['low']
                df.at[idx, 'volume'] = rt['volume']
                df.at[idx, 'turn'] = rt['turn']
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
        df['MA20'] = df['close'].rolling(20).mean()
        risk_level = self.calc_risk_level(curr['close'], df['MA5'].iloc[-1], df['MA20'].iloc[-1])

        signal_tags = []
        priority = 0
        action = "WAIT (观望)"

        is_3_up = all(df['pctChg'].tail(3) > 0)
        sum_3_rise = df['pctChg'].tail(3).sum()
        if (is_3_up and sum_3_rise <= 5 and winner_rate > 62):
            signal_tags.append("🔴温和吸筹")
            priority = max(priority, 60)
            action = "BUY (低吸)"

        turn_val = df['turn'].iloc[-1] 
        prev_turn = df['turn'].iloc[-2]
        is_high_turn = (turn_val > 5 and prev_turn > 5)

        if is_high_turn and winner_rate > 70:
            signal_tags.append("🔥换手锁仓")
            priority = max(priority, 70)
            action = "BUY (博弈)"

        df_60 = df.tail(60)
        limit_up_60 = len(df_60[df_60['pctChg'] > 9.5])
        if limit_up_60 >= 3 and winner_rate > 80 and days_listed > 30:
            signal_tags.append("🐲妖股基因")
            priority = max(priority, 90)
            action = "STRONG BUY"

        recent_20 = df.tail(20)
        has_limit_up_20 = len(recent_20[recent_20['pctChg'] > 9.5]) > 0
        has_gap = False
        recent_10 = df.tail(10).reset_index(drop=True)
        for i in range(1, len(recent_10)):
            if recent_10.iloc[i]['low'] > recent_10.iloc[i-1]['high']:
                has_gap = True; break
        is_red_15 = (df['close'].tail(15) > df['open'].tail(15)).astype(int)
        has_streak = (is_red_15.rolling(window=4).sum() == 4).any()
        vol_ma5 = df['volume'].tail(6).iloc[:-1].mean()
        is_double_vol = (curr['volume'] > prev['volume'] * 1.8) or (curr['volume'] > vol_ma5 * 1.8)

        if has_limit_up_20 and has_gap and has_streak and is_double_vol:
            signal_tags.append("👑四星共振")
            priority = 100
            action = "STRONG BUY"
        elif prev['open'] < prev['close'] and curr['close'] > prev['close']: 
             if priority == 0: 
                 action = "HOLD (持有)"
                 priority = 10
                 signal_tags.append("📈多头")

        if priority == 0: return None

        return {
            "result": {
                "代码": code, "名称": info['name'], "行业": info['industry'], 
                "现价": curr['close'], "涨跌": f"{curr['pctChg']:.2f}%", 
                "获利筹码": winner_rate, "风险评级": risk_level,
                "策略信号": " + ".join(signal_tags),
                "综合评级": action,
                "priority": priority
            },
            "alert": f"{info['name']}" if priority >= 90 else None,
            "option": f"{code} | {info['name']}"
        }

    def scan_market_optimized(self, code_list, max_price=None):
        results, alerts, valid_codes_list = [], [], []
        lg = bs.login()
        if lg.error_code != '0':
            st.error("连接服务器失败，请检查网络！")
            return [], [], []

        progress_bar = st.progress(0, text=f"🚀 正在启动稳定扫描 (共 {len(code_list)} 只)...")
        total = len(code_list)
        
        for i, code in enumerate(code_list):
            if i % 2 == 0:
                progress_bar.progress((i + 1) / total, text=f"🔍 正在分析: {code} ({i+1}/{total}) | 已命中: {len(results)} 只")
            try:
                res = self._process_single_stock(code, max_price)
                if res:
                    results.append(res["result"])
                    if res["alert"]: alerts.append(res["alert"])
                    valid_codes_list.append(res["option"])
            except:
                bs.logout(); time.sleep(0.5); bs.login()
                continue

        bs.logout()
        progress_bar.empty()
        return results, alerts, valid_codes_list

    # 🔥🔥🔥 核心修复：get_deep 增加 3 次自动重试 🔥🔥🔥
    @st.cache_data(ttl=600)
    def get_deep(_self, code):
        for i in range(3): # 重试3次
            bs.login()
            try:
                end = datetime.datetime.now().strftime("%Y-%m-%d")
                start = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
                rs = bs.query_history_k_data_plus(code, "date,open,close,high,low,volume,peTTM,pctChg", start_date=start, end_date=end, frequency="d", adjustflag="3")
                data = [r for r in rs.get_data()]
                bs.logout()
                if data: 
                    # 成功获取数据，立即返回
                    return pd.DataFrame(data, columns=["date", "open", "close", "high", "low", "volume", "peTTM", "pctChg"]).apply(pd.to_numeric, errors='coerce').dropna()
                else:
                    time.sleep(0.5) # 失败了歇会儿再试
            except:
                bs.logout()
                time.sleep(0.5)
        
        return None # 3次都失败才返回None

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

st.sidebar.header("🕹️ 控制台")
max_price_limit = st.sidebar.slider("💰 价格上限 (元)", 3.0, 100.0, 20.0)
pool_mode = st.sidebar.radio("🔎 选股范围:", ("中证500 (中小盘)", "沪深300 (大盘)", "手动输入"))
scan_limit = st.sidebar.slider("🔢 扫描数量 (池大小)", 50, 500, 200, step=50)

if pool_mode == "手动输入":
    default_pool = "600519, 002131, 002312, 600580, 002594"
    target_pool_str = st.sidebar.text_area("监控股票池", default_pool, height=100)
    final_code_list = target_pool_str.replace("，", ",").split(",")
else:
    if st.sidebar.button(f"📥 加载 {pool_mode} 成分股"):
        with st.spinner("正在获取成分股..."):
            index_code = "zz500" if "中证500" in pool_mode else "hs300"
            stock_list = engine.get_index_stocks(index_code)
            st.session_state['full_pool'] = stock_list 
            st.sidebar.success(f"已加载全量 {len(stock_list)} 只股票")
    
    if 'full_pool' in st.session_state:
        full_list = st.session_state['full_pool']
        final_code_list = full_list[:scan_limit] 
        st.sidebar.info(f"池内待扫: {len(final_code_list)} 只 (总库: {len(full_list)})")
    else:
        final_code_list = []

st.sidebar.markdown("---")
if st.sidebar.button("🚀 启动全策略扫描 (V65修复版)", type="primary"):
    if not final_code_list:
        st.sidebar.error("请先加载股票！")
    else:
        st.caption(f"当前筛选：价格 < {max_price_limit}元 | 剔除ST/科创/北交 | 模式：长连接稳定扫描")
        scan_res, alerts, valid_options = engine.scan_market_optimized(final_code_list, max_price=max_price_limit)
        st.session_state['scan_res'] = scan_res
        st.session_state['valid_options'] = valid_options
        st.session_state['alerts'] = alerts

with st.expander("📖 **策略逻辑白皮书**", expanded=False):
    st.markdown("##### 🔍 核心策略定义")
    for k, v in STRATEGY_LOGIC.items(): st.markdown(f"- **{k}**: {v}")

st.subheader(f"⚡ 扫描结果 (价格 < {max_price_limit}元)")

if 'scan_res' in st.session_state and st.session_state['scan_res']:
    results = st.session_state['scan_res']
    alerts = st.session_state.get('alerts', [])
    
    if alerts: 
        alert_names = "、".join(alerts)
        st.success(f"🔥 发现 {len(alerts)} 只【主力高控盘】标的：**{alert_names}**")
    
    df_scan = pd.DataFrame(results).sort_values(by="priority", ascending=False)
    
    if df_scan.empty:
        st.warning(f"⚠️ 扫描完成，无符合条件的股票。")
    else:
        st.dataframe(
            df_scan, use_container_width=True, hide_index=True,
            column_config={
                "代码": st.column_config.TextColumn("代码"),
                "名称": st.column_config.TextColumn("名称"),
                "获利筹码": st.column_config.ProgressColumn("获利筹码(%)", format="%.1f%%", min_value=0, max_value=100),
                "风险评级": st.column_config.TextColumn("风险评级", help="基于乖离率计算"),
                "策略信号": st.column_config.TextColumn("策略信号", help=STRATEGY_TIP, width="large"),
                "综合评级": st.column_config.TextColumn("综合评级", help=ACTION_TIP, width="medium"),
                "priority": None
            }
        )
else:
    st.info("👈 请在左侧加载股票 -> 点击“启动全策略扫描”")

st.divider()

if 'valid_options' in st.session_state and st.session_state['valid_options']:
    st.subheader("🧠 深度分析")
    target = st.selectbox("选择目标进行深度分析", st.session_state['valid_options'])
    
    target_code = target.split("|")[0].strip()
    target_name = target.split("|")[1].strip()

    if st.button(f"🚀 立即分析 {target_name}"):
        with st.spinner("AI 正在深度运算..."):
            
            # 🔥 尝试获取数据
            df = engine.get_deep(target_code)
            rt = engine.get_realtime_quote(target_code)
            
            if df is not None:
                if rt:
                    if str(df.iloc[-1]['date']) != str(rt['date']):
                         new = pd.DataFrame([{"date":rt['date'], "open":rt['open'], "close":rt['close'], "high":rt['high'], "low":rt['low'], "volume":rt['volume'], "peTTM":0, "pctChg": 0}])
                         df = pd.concat([df, new], ignore_index=True)
                    else:
                         df.at[df.index[-1], 'close'] = rt['close']

                df = engine.calc_indicators(df)
                future_info = engine.run_ai_prediction(df)
                
                if future_info:
                    last = df.iloc[-1]
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("当前价格", f"¥{last['close']:.2f}")
                    col2.metric("AI预测明日", f"¥{future_info['pred_price']:.2f}", delta=f"{future_info['pred_price']-last['close']:.2f}", delta_color="inverse")
                    pe = last.get('peTTM', 0)
                    col3.metric("PE估值", f"{pe:.1f}")
                    
                    if future_info['color'] == 'red':
                        st.error(f"### {future_info['title']}\n{future_info['desc']}\n\n**{future_info['action']}**")
                    elif future_info['color'] == 'green':
                        st.success(f"### {future_info['title']}\n{future_info['desc']}\n\n**{future_info['action']}**")
                    else:
                        st.info(f"### {future_info['title']}\n{future_info['desc']}\n\n**{future_info['action']}**")

                    st.markdown("### 📅 AI 时空推演 (未来3日)")
                    d_cols = st.columns(3)
                    for i in range(3):
                        d_cols[i].metric(label=future_info['dates'][i], value=f"¥{future_info['prices'][i]:.2f}", delta="预测", delta_color="inverse")

                    fig = engine.plot_professional_kline(df, target_name)
                    st.plotly_chart(fig, use_container_width=True)
                    st.info("💡 **图例**: 🔺红色B=金叉买点 | 🔻绿色S=死叉卖点 (仅供辅助参考)")
                else:
                    st.warning("数据不足，无法预测")
            else:
                # 🔥 V65: 这里会弹出报错，而不是沉默
                st.error("❌ 数据获取失败（网络超时或接口繁忙）。请等待 5 秒后再次点击按钮！")

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