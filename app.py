import streamlit as st
import baostock as bs
import pandas as pd
import numpy as np
import time
import datetime
from sklearn.linear_model import LinearRegression
import concurrent.futures
import threading  # 引入锁机制

# ==========================================
# 0. 全局配置与安全锁
# ==========================================
# ⚠️ 核心修改：创建一个全局锁，防止多线程同时抢占 Baostock 导致崩溃
bs_lock = threading.Lock()

STRATEGY_TIP = """
🚀 金叉突变: 短期均线向上突破长期均线，建议买入
⚡ 死叉破位: 短期均线向下跌破长期均线，建议卖出
📈 多头持仓: 均线发散向上，处于上升通道，建议持有
📉 空仓回避: 均线发散向下，处于下跌通道，建议空仓
⚪ 震荡观望: 均线纠缠，方向不明，建议观望
"""

# ==========================================
# 1. 核心引擎 (加锁优化版)
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

    # --- 🧵 线程工作函数 ---
    def _process_single_stock(self, code):
        code = self.clean_code(code)
        
        # ⚠️ 核心优化：只取最近 40 天数据 (算 MA20 足够了)，大幅减少网络传输时间
        end = datetime.datetime.now().strftime("%Y-%m-%d")
        start = (datetime.datetime.now() - datetime.timedelta(days=60)).strftime("%Y-%m-%d")
        
        data = []
        name = code
        
        # 🔥 关键点：使用 with bs_lock 确保这一段网络请求是安全的
        # 虽然这里变成了串行，但因为不需要重复登录，整体速度比 V15 快 10 倍以上
        with bs_lock:
            try:
                # 1. 获取名字
                rs_name = bs.query_stock_basic(code=code)
                if rs_name.error_code == '0' and rs_name.next():
                    name = rs_name.get_row_data()[1]
                
                # 2. 过滤
                if not self.is_valid(code, name):
                    return None

                # 3. 获取K线
                rs = bs.query_history_k_data_plus(code, "date,close,volume,pctChg", start_date=start, frequency="d", adjustflag="3")
                while rs.next(): 
                    data.append(rs.get_row_data())
            except:
                return None

        # --- 以下计算逻辑在锁外面执行，享受多线程加速 ---
        if not data: return None
        
        df = pd.DataFrame(data, columns=["date", "close", "volume", "pctChg"])
        df[['close','volume','pctChg']] = df[['close','volume','pctChg']].astype(float)
        
        if len(df) < 20: return None # 数据太少算不了均线

        df['MA5'] = df['close'].rolling(5).mean()
        df['MA20'] = df['close'].rolling(20).mean()
        df['Vol_MA5'] = df['volume'].rolling(5).mean()
        
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        signal = "⚪ 震荡观望"
        action = "WAIT (等待)"
        priority = 0
        
        # 信号判断逻辑 (保持不变)
        if prev['MA5'] <= prev['MA20'] and curr['MA5'] > curr['MA20']:
            if curr['volume'] > curr['Vol_MA5'] * 1.5:
                signal = "🚀 金叉突变 (放量)"
                action = "BUY (建议买入)"
                priority = 10
            else:
                signal = "🚀 金叉突变"
                action = "BUY (建议买入)"
                priority = 9
        elif prev['MA5'] >= prev['MA20'] and curr['MA5'] < curr['MA20']:
            signal = "⚡ 死叉破位"
            action = "SELL (建议卖出)"
            priority = 8
        elif curr['close'] > curr['MA5'] > curr['MA20']:
            signal = "📈 多头持仓"
            action = "HOLD (多头持有)"
            priority = 5
        elif curr['close'] < curr['MA5'] < curr['MA20']:
            signal = "📉 空仓回避"
            action = "AVOID (建议空仓)"
            priority = 1
            
        return {
            "result": {
                "代码": code, "名称": name, "现价": f"¥{curr['close']:.2f}", 
                "涨跌幅": f"{curr['pctChg']:.2f}%", 
                "策略信号": signal, "操作建议": action, "priority": priority
            },
            "alert": name if priority >= 9 else None,
            "option": f"{code} | {name}"
        }

    def scan_market_optimized(self, code_list):
        results, alerts, valid_codes_list = [], [], []
        
        # ⚠️ 优化：在主线程统一登录一次，极其高效
        lg = bs.login()
        if lg.error_code != '0':
            return [], [], []

        progress_bar = st.progress(0, text="正在启动极速安全扫描...")
        total = len(code_list)
        
        # 开启 8 线程
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                future_to_code = {executor.submit(self._process_single_stock, c): c for c in code_list}
                
                for i, future in enumerate(concurrent.futures.as_completed(future_to_code)):
                    progress_bar.progress((i + 1) / total, text=f"正在分析 {i+1}/{total} ...")
                    try:
                        res = future.result()
                        if res:
                            results.append(res["result"])
                            if res["alert"]: alerts.append(res["alert"])
                            valid_codes_list.append(res["option"])
                    except:
                        continue
        finally:
            # 确保最后一定会退出登录
            bs.logout()
            progress_bar.empty()
            
        return results, alerts, valid_codes_list

    @st.cache_data(ttl=600)
    def get_deep_data(_self, code):
        """深度数据获取 (保持独立连接，防止干扰)"""
        # 这里单独Login一次没关系，因为用户点击频率低
        bs.login()
        try:
            end = datetime.datetime.now().strftime("%Y-%m-%d")
            start = (datetime.datetime.now() - datetime.timedelta(days=400)).strftime("%Y-%m-%d")
            rs = bs.query_history_k_data_plus(code, "date,close,high,low,volume,peTTM,pbMRQ", start_date=start, end_date=end, frequency="d", adjustflag="3")
            data = []
            while rs.next(): data.append(rs.get_row_data())
            if not data: return None
            df = pd.DataFrame(data, columns=["date", "close", "high", "low", "volume", "peTTM", "pbMRQ"])
            cols = ['close', 'high', 'low', 'volume', 'peTTM', 'pbMRQ']
            df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
            df.dropna(subset=['close'], inplace=True)
            return df
        finally:
            bs.logout()

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
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['std'] = df['close'].rolling(20).std()
        df['upper'] = df['MA20'] + 2 * df['std']
        df['lower'] = df['MA20'] - 2 * df['std']
        return df

# ==========================================
# 2. 界面 UI (A股配色 + 全功能)
# ==========================================
st.set_page_config(page_title="V16 工业级优化版", layout="wide", page_icon="⚡")
engine = QuantsEngine()

st.sidebar.header("🕹️ 控制台")
auto_refresh = st.sidebar.checkbox("⏱️ 开启自动刷新", value=False)
refresh_rate = st.sidebar.slider("刷新频率 (秒)", 5, 60, 15)
st.sidebar.markdown("---")

default_pool = "600519, 601318, 000858, 600580, 002594, 300750, 600036"
user_pool = st.sidebar.text_area("📋 监控股票池 (逗号分隔)", default_pool, height=100)

# --- 执行优化的扫描 ---
pool_list = user_pool.replace("，", ",").split(",")
if user_pool:
    scan_res, alerts, valid_options = engine.scan_market_optimized(pool_list)
else:
    scan_res, alerts, valid_options = [], [], []

st.sidebar.markdown("---")
st.sidebar.markdown("👇 **深度分析选择**")

select_options = valid_options if valid_options else ["sh.600519 | 贵州茅台"]
selected_option = st.sidebar.selectbox("🔍 选择目标", select_options)
target_code = selected_option.split("|")[0].strip()

# --- 主界面 ---
with st.expander("📖 **新手必读：策略信号说明书**", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    c1.error("**🚀 金叉突变**\n\n短期均线向上突破长期均线，主力资金介入，**买入信号**。")
    c2.success("**⚡ 死叉破位**\n\n短期均线跌破长期均线，上升趋势结束，**卖出信号**。")
    c3.error("**📈 多头持仓**\n\n股价沿着均线稳步上涨，趋势健康，**建议持有**。")
    c4.success("**📉 空仓回避**\n\n股价持续下跌，切勿盲目抄底，**建议空仓**。")

st.subheader("⚡ 实盘信号雷达 (安全极速版)")

if alerts:
    st.error(f"🔔 **突发警报**：监测到 {len(alerts)} 只股票出现【金叉买入】信号！-> {', '.join(alerts)}")
    st.toast(f"发现买入机会：{alerts[0]}", icon="🚀")

if scan_res:
    df_scan = pd.DataFrame(scan_res).sort_values(by="priority", ascending=False)
    st.dataframe(
        df_scan, use_container_width=True, hide_index=True,
        column_config={
            "代码": st.column_config.TextColumn("代码"),
            "名称": st.column_config.TextColumn("名称"),
            "现价": st.column_config.TextColumn("现价"),
            "涨跌幅": st.column_config.TextColumn("涨跌幅"),
            "策略信号": st.column_config.TextColumn("策略信号", help=STRATEGY_TIP, width="medium"),
            "操作建议": st.column_config.TextColumn("操作建议", width="medium"),
            "priority": None
        }
    )
else:
    st.info("监控池正在初始化...")

st.divider()

st.subheader(f"🧠 AI 深度分析指挥部: {selected_option}")

if st.button(f"🚀 立即分析 {target_code}", type="primary"):
    with st.spinner(f"正在挖掘 {target_code} 数据..."):
        df = engine.get_deep_data(target_code)
    
    if df is not None:
        df = engine.calc_indicators(df)
        pred_price = engine.run_ai_prediction(df)
        last = df.iloc[-1]
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("当前价格", f"¥{last['close']:.2f}")
        
        # A股配色：上涨/正数=红色(inverse)，下跌/负数=绿色
        c2.metric("AI预测明日", f"¥{pred_price:.2f}", delta=f"{pred_price - last['close']:.2f}", delta_color="inverse")
        
        pe_val = last.get('peTTM', np.nan)
        if pd.isna(pe_val): pe_str, pe_status = "暂无数据", "未知"
        else:
            avg_pe = df['peTTM'].mean()
            pe_str = f"{pe_val:.1f}"
            pe_status = "低估" if pe_val < avg_pe else "高估"
            
        c3.metric("PE估值", pe_str, delta=pe_status, delta_color="off")
        
        score = 50
        if pred_price > last['close']: score += 10
        if pe_status == "低估": score += 15
        if last['MACD'] > 0: score += 10
        if last['close'] > last['MA20']: score += 10
        if last['RSI'] < 20: score += 15
        
        c4.metric("AI综合评分", f"{score} 分")
        
        with st.expander("📋 **点击查看详细技术分析报告**", expanded=True):
            if last['DIF'] > last['DEA']: st.markdown("✅ **MACD**: 处于多头区域 (金叉状态)。")
            else: st.markdown("⚠️ **MACD**: 处于空头区域 (死叉状态)。")
            if last['close'] < last['lower']: st.markdown("💎 **布林带**: 股价跌破下轨，**超跌反弹**机会！")
            if pe_status != "未知":
                st.markdown(f"🏢 **基本面**: 当前市盈率 {pe_str}，历史平均 {avg_pe:.1f}，处于 **{pe_status}** 区间。")

        t1, t2 = st.tabs(["📊 价格预测 & 布林带", "📈 MACD & RSI 趋势"])
        with t1:
            st.line_chart(df.set_index('date')[['close', 'MA20', 'upper', 'lower']], color=["#000000", "#FF0000", "#CCCCCC", "#CCCCCC"])
        with t2:
            st.line_chart(df.set_index('date')[['MACD', 'RSI']])
    else:
        st.error(f"❌ 无法获取 {target_code} 的数据。")

if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()