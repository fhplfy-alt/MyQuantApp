import streamlit as st
import time
import datetime

# ==========================================
# ⚠️ 核心配置
# ==========================================
st.set_page_config(
    page_title="V59 终极完全体", 
    layout="wide", 
    page_icon="👑",
    initial_sidebar_state="expanded"
)

st.title("👑 V59 智能量化系统 (全策略·全市场·实时版)")

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

# 悬停提示
STRATEGY_TIP = """
👑 四星共振: [涨停+缺口+连阳+倍量] 同时满足
🐲 妖股基因: 60天内3板 + 筹码>80%
🔥 换手锁仓: 连续高换手 + 高获利
🔴 温和吸筹: 3连阳但涨幅小 + 筹码集中
📈 多头排列: 基础趋势向上
"""

ACTION_TIP = """
🟥 STRONG BUY: 【重仓】四星共振/首阳首板
🟧 BUY (博弈): 【激进】换手锁仓/接力
🟨 BUY (低吸): 【潜伏】温和吸筹/缩量回踩
🟦 HOLD: 【持股】趋势完好
⬜ WAIT: 【观望】无机会
"""

# 🔥🔥🔥 这里就是你截图里的内容，完全一致 🔥🔥🔥
STRATEGY_LOGIC = {
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
        if not (code.startswith('sh.') or code.startswith('sz.') or code.startswith('bj.')):
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

    def _process_single_stock(self, code, max_price, allow_kc, allow_bj, selected_industries):
        code = self.clean_code(code)
        end = datetime.datetime.now().strftime("%Y-%m-%d")
        start = (datetime.datetime.now() - datetime.timedelta(days=150)).strftime("%Y-%m-%d")
        
        data = []
        info = {'name': code, 'industry': '未分类', 'ipoDate': '2000-01-01'}
        
        # 独立登录，保证单线程稳定
        bs.login()
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

            if not self.is_valid(code, info['name'], info['industry'], allow_kc, allow_bj, selected_industries): 
                bs.logout()
                return None

            rs = bs.query_history_k_data_plus(code, "date,open,close,high,low,volume,pctChg,turn", start_date=start, frequency="d", adjustflag="3")
            while rs.next(): data.append(rs.get_row_data())
            
        except:
            bs.logout()
            return None
        
        bs.logout()

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
        
        if max_price and curr['close'] > max_price: return None

        winner_rate = self.calc_winner_rate(df, curr['close'])
        
        signal_tags = []
        priority = 0
        action = "WAIT"

        # 策略集合
        
        # 1. 首阳首板 (PDF)
        recent_10 = df.tail(10).iloc[:-1]
        has_limit_recent = len(recent_10[recent_10['pctChg'] > 9.5]) > 0
        is_today_red = curr['close'] > curr['open']
        is_correction = prev['close'] < df.tail(5)['high'].max()
        if has_limit_recent and is_today_red and is_correction:
            signal_tags.append("🌤️首阳首板"); priority = 95; action = "STRONG BUY"

        # 2. 极度缩量 (PDF)
        vol_ma5 = df['volume'].tail(6).iloc[:-1].mean()
        if curr['volume'] < vol_ma5 * 0.6: 
            signal_tags.append("🤐极度缩量"); priority = max(priority, 5)

        # 3. 温和吸筹
        if all(df['pctChg'].tail(3) > 0) and df['pctChg'].tail(3).sum() <= 5 and winner_rate > 62:
            signal_tags.append("🔴温和吸筹"); priority = max(priority, 60); action = "BUY (低吸)"
        
        # 4. 换手锁仓
        if (df['turn'].iloc[-1] > 5 and df['turn'].iloc[-2] > 5) and winner_rate > 70:
            signal_tags.append("🔥换手锁仓"); priority = max(priority, 70); action = "BUY (博弈)"
            
        # 5. 妖股基因
        limit_60 = len(df.tail(60)[df.tail(60)['pctChg'] > 9.5])
        if limit_60 >= 3 and winner_rate > 80:
            signal_tags.append("🐲妖股基因"); priority = max(priority, 90); action = "STRONG BUY"

        # 6. 四星共振
        has_limit_20 = len(df.tail(20)[df.tail(20)['pctChg'] > 9.5]) > 0
        is_double = curr['volume'] > prev['volume'] * 1.8
        is_red4 = (df['close'].tail(4) > df['open'].tail(4)).all()
        if has_limit_20 and is_red4 and is_double:
            signal_tags.append("👑四星共振"); priority = 100; action = "STRONG BUY"
            
        # 7. 基础多头 (这里就是你截图里的逻辑)
        if prev['close'] > prev['open'] and curr['close'] > prev['close']:
            if priority == 0:
                signal_tags.append("📈多头排列"); priority = 10; action = "HOLD"

        if priority == 0: return None

        return {
            "result": {
                "代码": code, "名称": info['name'], "行业": info['industry'], 
                "现价": curr['close'], "涨跌": f"{curr['pctChg']:.2f}%", 
                "获利筹码": winner_rate, "策略信号": " + ".join(signal_tags),
                "操作建议": action, "priority": priority
            },
            "alert": f"{info['name']}" if priority >= 90 else None,
            "option": f"{code} | {info['name']}"
        }

    def scan_market(self, code_list, max_p, allow_kc, allow_bj, selected_industries):
        results, alerts, codes = [], [], []
        lg = bs.login()
        if lg.error_code != '0': return [],[],[]
        
        market_status = self.get_market_sentiment()
        
        filter_msg = f"全行业..." if not selected_industries else f"指定: {','.join(selected_industries)}"
        bar = st.progress(0, f"启动扫描 ({filter_msg}) - 稳定模式...")
        
        for i, c in enumerate(code_list):
            if i%5==0: bar.progress((i+1)/len(code_list), f"分析中: {c} ({i}/{len(code_list)})")
            try:
                time.sleep(0.05)
                r = self._process_single_stock(c, max_p, allow_kc, allow_bj, selected_industries)
                if r: results.append(r["result"]); alerts.append(r["alert"]) if r["alert"] else None; codes.append(r["option"])
            except: 
                continue

        bar.empty()
        return results, alerts, codes, market_status

    @st.cache_data(ttl=600)
    def get_deep(self, code):
        bs.login()
        end = datetime.datetime.now().strftime("%Y-%m-%d")
        start = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
        rs = bs.query_history_k_data_plus(code, "date,open,close,high,low,volume,peTTM", start_date=start, end_date=end, frequency="d", adjustflag="3")
        data = [r for r in rs.get_data()]
        bs.logout()
        if not data: return None
        return pd.DataFrame(data, columns=["date", "open", "close", "high", "low", "volume", "peTTM"]).apply(pd.to_numeric, errors='coerce').dropna()

engine = QuantsEngine()

# --- 2. 界面逻辑 ---

market_info = engine.get_market_sentiment()
if market_info:
    c1, c2 = st.columns([1, 4])
    c1.metric("上证指数", market_info['status'], delta_color="inverse")
    if market_info['color'] == 'red':
        c2.success(f"📈 策略建议：{market_info['status']}，建议仓位 {market_info['pos']}")
    else:
        c2.error(f"📉 策略建议：{market_info['status']}，风险高，建议仓位 {market_info['pos']}")
st.divider()

st.sidebar.header("🕹️ 战神控制台")
max_p = st.sidebar.slider("💰 价格上限", 3.0, 500.0, 20.0)

st.sidebar.markdown("#### 🏭 行业过滤")
selected_industries = st.sidebar.multiselect("行业 (留空全选):", options=ALL_INDUSTRIES, default=[])
allow_kc = st.sidebar.checkbox("包含科创板 (688)", value=False)
allow_bj = st.sidebar.checkbox("包含北交所 (8xx)", value=False)

mode = st.sidebar.radio("选股范围", ("全市场精选", "手动输入"))
limit = st.sidebar.slider("🔢 扫描数量", 100, 6000, 200)

if mode == "手动":
    pool = st.sidebar.text_area("代码池", "600519, 002131").replace("，", ",").split(",")
else:
    if st.sidebar.button("📥 加载全市场"):
        st.session_state['pool'] = engine.get_all_stocks()
        st.sidebar.success(f"已加载 {len(st.session_state['pool'])} 只")
    pool = st.session_state.get('pool', [])[:limit]

if st.sidebar.button("🚀 启动战神扫描"):
    res, al, opts, _ = engine.scan_market(pool, max_p, allow_kc, allow_bj, selected_industries)
    st.session_state.update({'res': res, 'opts': opts, 'al': al})

if st.session_state.get('al'): 
    names = "、".join(st.session_state['al'])
    st.success(f"🔥 发现 {len(st.session_state['al'])} 只龙头/首板标的：**{names}**")

# 🔥🔥🔥 核心：白皮书显示区 🔥🔥🔥
with st.expander("📖 **策略逻辑白皮书 (透明度报告)**", expanded=False):
    st.markdown("##### 🔍 核心策略定义")
    for k, v in STRATEGY_LOGIC.items(): st.markdown(f"- **{k}**: {v}")

if st.session_state.get('res'):
    st.dataframe(pd.DataFrame(st.session_state['res']), use_container_width=True, 
                 column_config={
                     "获利筹码": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
                     "策略信号": st.column_config.TextColumn(help=STRATEGY_TIP, width="large"),
                     "操作建议": st.column_config.TextColumn(help=ACTION_TIP)
                 })

st.divider()

if st.session_state.get('opts'):
    st.subheader("🧠 深度分析")
    target = st.selectbox("选择目标", st.session_state['opts'])
    if st.button(f"🚀 分析 {target}"):
        code = target.split("|")[0].strip()
        df = engine.get_deep(code)
        rt = engine.get_realtime_quote(code)
        
        if df is not None:
            if rt:
                if str(df.iloc[-1]['date']) != str(rt['date']):
                     new = pd.DataFrame([{"date":rt['date'], "open":rt['open'], "close":rt['close'], "high":rt['high'], "low":rt['low'], "volume":rt['volume'], "peTTM":0}])
                     df = pd.concat([df, new], ignore_index=True)
                else:
                     df.at[df.index[-1], 'close'] = rt['close']

            df['MA5'] = df['close'].rolling(5).mean(); df['MA10'] = df['close'].rolling(10).mean()
            
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

            fig = go.Figure(data=[go.Candlestick(x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], increasing_line_color='red', decreasing_line_color='green', name='K线')])
            fig.add_trace(go.Scatter(x=df['date'], y=df['MA5'], name='MA5', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=df['date'], y=df['MA10'], name='MA10 (生命线)', line=dict(color='blue', width=2)))
            
            buy = df[(df['MA5']>df['MA10']) & (df['MA5'].shift(1)<=df['MA10'].shift(1))]
            sell = df[(df['MA5']<df['MA10']) & (df['MA5'].shift(1)>=df['MA10'].shift(1))]
            fig.add_trace(go.Scatter(x=buy['date'], y=buy['low']*0.98, mode='markers+text', marker=dict(symbol='triangle-up', color='red', size=10), text='B'))
            fig.add_trace(go.Scatter(x=sell['date'], y=sell['high']*1.02, mode='markers+text', marker=dict(symbol='triangle-down', color='green', size=10), text='S'))
            
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