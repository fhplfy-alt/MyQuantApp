import streamlit as st
import plotly.graph_objects as go

# ==========================================
# ⚠️ 核心配置
# ==========================================
st.set_page_config(
    page_title="V36 终极体验版", 
    layout="wide", 
    page_icon="🧬",
    initial_sidebar_state="expanded"
)

st.title("🧬 V36 智能量化系统 (交互体验增强版)")

import baostock as bs
import pandas as pd
import numpy as np
import time
import datetime
from sklearn.linear_model import LinearRegression
import concurrent.futures
import threading

# ==========================================
# 0. 全局配置与悬停文案 (这里就是鼠标放上去显示的内容)
# ==========================================
bs_lock = threading.Lock()

# 📝 策略信号的悬停说明
STRATEGY_TIP = """
👇 信号含义说明：
👑 四星共振: [涨停+缺口+连阳+倍量] 同时满足，最强主升浪信号！
🐲 妖股基因: 60天内3板 + 筹码>80%，游资龙头特征。
🔥 换手锁仓: 连续高换手 + 高获利，主力接力迹象。
🔴 温和吸筹: 3连阳但涨幅小 + 筹码集中，主力潜伏期。
🚀 金叉/多头: 基础均线趋势向上。
"""

# 📝 综合评级的悬停说明
ACTION_TIP = """
👇 操作建议说明：
🟥 STRONG BUY: 【重点关注】确定性极高，适合重仓 (如四星/妖股)。
🟧 BUY (博弈): 【激进买入】适合短线快进快出，博取连板。
🟨 BUY (低吸): 【稳健买入】主力吸筹期，适合逢低建仓。
🟦 HOLD: 【持股】趋势完好，拿住不动。
⬜ WAIT: 【观望】无机会或风险大。
"""

# 策略逻辑字典
STRATEGY_LOGIC = {
    "👑 四星共振": "近20日有涨停 + 向上跳空缺口 + 4连阳 + 量比>1.8",
    "🐲 妖股基因": "近60日涨停≥3次 + 获利筹码>80% + 上市>30天",
    "🔥 换手锁仓": "连续2日换手率>5% + 获利筹码>70%",
    "🔴 温和吸筹": "3连阳且累计涨幅<5% + 获利筹码>62%",
    "⚠️ 风险评级": "基于乖离率(BIAS)评估"
}

# ==========================================
# 1. 核心引擎 (保持不变)
# ==========================================
class QuantsEngine:
    def __init__(self):
        pass

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

    def get_index_stocks(self, index_type="zz500"):
        bs.login()
        stocks = []
        try:
            if index_type == "hs300": rs = bs.query_hs300_stocks()
            else: rs = bs.query_zz500_stocks()
            while rs.next(): stocks.append(rs.get_row_data()[1])
        except: pass
        finally: bs.logout()
        return stocks

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

    def _process_single_stock(self, code, max_price=None):
        code = self.clean_code(code)
        end = datetime.datetime.now().strftime("%Y-%m-%d")
        start = (datetime.datetime.now() - datetime.timedelta(days=150)).strftime("%Y-%m-%d")
        
        data = []
        info = {'name': code, 'industry': '-', 'ipoDate': '2000-01-01'}
        
        with bs_lock:
            for attempt in range(3):
                try:
                    rs_info = bs.query_stock_basic(code=code)
                    if rs_info.error_code != '0': raise Exception("Lost")
                    if rs_info.next():
                        row = rs_info.get_row_data()
                        info['name'] = row[1]
                        info['ipoDate'] = row[2]
                    rs_ind = bs.query_stock_industry(code)
                    if rs_ind.error_code == '0' and rs_ind.next():
                        info['industry'] = rs_ind.get_row_data()[3] 
                    if not self.is_valid(code, info['name']): return None
                    rs = bs.query_history_k_data_plus(code, "date,open,close,high,low,volume,pctChg,turn", start_date=start, frequency="d", adjustflag="3")
                    if rs.error_code != '0': raise Exception("Data Fail")
                    while rs.next(): data.append(rs.get_row_data())
                    time.sleep(0.01)
                    break 
                except:
                    bs.logout(); time.sleep(0.5); bs.login()

        if not data: return None
        try:
            df = pd.DataFrame(data, columns=["date", "open", "close", "high", "low", "volume", "pctChg", "turn"])
            df = df.apply(pd.to_numeric, errors='coerce')
        except: return None
        if len(df) < 60: return None

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

        # --- 策略逻辑 ---
        signal_tags = []
        priority = 0
        action = "WAIT (观望)"

        is_3_up = all(df['pctChg'].tail(3) > 0)
        sum_3_rise = df['pctChg'].tail(3).sum()
        if (is_3_up and sum_3_rise <= 5 and winner_rate > 62):
            signal_tags.append("🔴温和吸筹")
            priority = max(priority, 60)
            action = "BUY (低吸)"

        is_high_turn = all(df['turn'].tail(2) > 5) 
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
                "代码": code, "名称": info['name'], 
                "所属行业": info['industry'],
                "现价": curr['close'], 
                "涨跌": f"{curr['pctChg']:.2f}%", 
                "获利筹码": winner_rate,
                "风险评级": risk_level,
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
        if lg.error_code != '0': return [], [], []
        progress_bar = st.progress(0, text=f"🔍 正在扫描 {len(code_list)} 只股票...")
        total = len(code_list)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                future_to_code = {executor.submit(self._process_single_stock, c, max_price): c for c in code_list}
                for i, future in enumerate(concurrent.futures.as_completed(future_to_code)):
                    if i % 5 == 0: progress_bar.progress((i + 1) / total, text=f"📊 扫描进度 {int((i+1)/total*100)}% | 命中: {len(results)} 只...")
                    try:
                        res = future.result()
                        if res:
                            results.append(res["result"])
                            if res["alert"]: alerts.append(res["alert"])
                            valid_codes_list.append(res["option"])
                    except: continue
        finally:
            bs.logout()
            progress_bar.empty()
        return results, alerts, valid_codes_list

    @st.cache_data(ttl=600)
    def get_deep_data(_self, code):
        bs.login()
        try:
            end = datetime.datetime.now().strftime("%Y-%m-%d")
            start = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
            rs = bs.query_history_k_data_plus(code, "date,open,close,high,low,volume,peTTM,pbMRQ", start_date=start, end_date=end, frequency="d", adjustflag="3")
            data = []
            while rs.next(): data.append(rs.get_row_data())
            if not data: return None
            df = pd.DataFrame(data, columns=["date", "open", "close", "high", "low", "volume", "peTTM", "pbMRQ"])
            cols = ['open', 'close', 'high', 'low', 'volume', 'peTTM', 'pbMRQ']
            df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
            df.dropna(subset=['close'], inplace=True)
            return df
        finally: bs.logout()

    def run_ai_prediction(self, df):
        if len(df) < 30: return 0
        recent = df.tail(30).reset_index(drop=True)
        X = np.array(recent.index).reshape(-1, 1)
        y = recent['close'].values
        model = LinearRegression()
        model.fit(X, y)
        return model.predict(np.array([[30]]))[0]

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
        if not sell.empty:
            fig.add_trace(go.Scatter(x=sell['date'], y=sell['high']*1.02, mode='markers+text', marker=dict(symbol='triangle-down', size=12, color='green'), text='S', textposition='top center', name='卖出'))

        fig.update_layout(title=f"{title} - 智能操盘K线 (含B/S点)", xaxis_rangeslider_visible=False, height=600)
        return fig

# ==========================================
# 2. 界面 UI
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
if st.sidebar.button("🚀 启动全策略扫描", type="primary"):
    if not final_code_list:
        st.sidebar.error("请先加载股票！")
    else:
        st.caption(f"当前筛选：价格 < {max_price_limit}元 | 剔除ST/科创/北交 | 扫描策略：四星+妖股+换手+吸筹")
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
        # 🔥🔥🔥 核心修改点：加入 help 参数 🔥🔥🔥
        st.dataframe(
            df_scan, use_container_width=True, hide_index=True,
            column_config={
                "代码": st.column_config.TextColumn("代码"),
                "名称": st.column_config.TextColumn("名称"),
                "获利筹码": st.column_config.ProgressColumn("获利筹码(%)", format="%.1f%%", min_value=0, max_value=100),
                "风险评级": st.column_config.TextColumn("风险评级", help="基于乖离率计算"),
                
                # 👇 这里加了 STRATEGY_TIP
                "策略信号": st.column_config.TextColumn("策略信号", help=STRATEGY_TIP, width="large"),
                
                # 👇 这里加了 ACTION_TIP
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
        with st.spinner("AI 正在绘制 B/S 点操盘图..."):
            df = engine.get_deep_data(target_code)
            if df is not None:
                df = engine.calc_indicators(df)
                pred = engine.run_ai_prediction(df)
                last = df.iloc[-1]
                
                col1, col2, col3 = st.columns(3)
                col1.metric("当前价格", f"¥{last['close']:.2f}")
                col2.metric("AI预测明日", f"¥{pred:.2f}", delta=f"{pred-last['close']:.2f}", delta_color="inverse")
                pe = last.get('peTTM', 0)
                col3.metric("PE估值", f"{pe:.1f}")
                
                fig = engine.plot_professional_kline(df, target_name)
                st.plotly_chart(fig, use_container_width=True)
                st.info("💡 **图例**: 🔺红色B=金叉买点 | 🔻绿色S=死叉卖点 (仅供辅助参考)")