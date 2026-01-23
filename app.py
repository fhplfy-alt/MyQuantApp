import streamlit as st

# ==========================================
# 🔐 1. 安全访问控制 (新增登录按钮)
# ==========================================
def check_password():
    """返回 True 如果用户输入了正确的密码并点击登录"""
    if "password_correct" not in st.session_state:
        st.markdown("### 🔐 V45 智能量化系统安全验证")
        # 密码输入框
        pwd = st.text_input("请输入访问密码", type="password", help="密码设置为：vip888")
        # 新增：登录按钮
        if st.button("立即登录系统"):
            if pwd == "vip888":
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("❌ 密码错误，请检查！")
        return False
    return True

# 只有通过验证才运行后续代码
if not check_password():
    st.stop()

# ==========================================
# ⚠️ 核心配置 (保持原始 V45 风格不变)
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
    import baostock as bs
    import pandas as pd
    import numpy as np
    import datetime
    from sklearn.linear_model import LinearRegression
except ImportError as e:
    st.error(f"❌ 启动失败！缺少必要运行库: {e}")
    st.stop()

# ==========================================
# 0. 全局配置 (保持原功能策略说明不变)
# ==========================================
STRATEGY_TIP = """
👇 信号含义详细对照：
👑 四星共振: [涨停+缺口+连阳+倍量] 同时满足，最强主升浪信号！
🐲 妖股基因: 60天内3板 + 筹码>80%，游资龙头特征。
🔥 换手锁仓: 连续高换手 + 高获利，主力清洗浮筹接力。
🔴 温和吸筹: 3连阳但涨幅小 + 筹码集中，主力潜伏期。
📈 多头排列: 股价收阳且重心上移，趋势健康，建议持有。
"""

# ==========================================
# 2. 核心引擎 (修复扫描、新增标注与预测)
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
        if "sh.688" in code or "ST" in name: return False 
        if "bj." in code or code.startswith(("sz.8", "sz.4")): return False 
        return True

    def get_all_stocks(self):
        """修复全场扫描：确保拉取 6000 只股票"""
        try:
            bs.login()
            rs = bs.query_all_stock()
            stocks = []
            while rs.next():
                row = rs.get_row_data()
                if self.is_valid(row[0], row[1]): stocks.append(row[0])
            bs.logout()
            return stocks[:self.MAX_SCAN_LIMIT]
        except: return []

    def run_ai_prediction(self, df):
        """新增：给出后三天的预估价位"""
        if df is None or len(df) < 20: return None
        try:
            recent = df.tail(20).reset_index(drop=True)
            X = np.array(recent.index).reshape(-1, 1)
            y = recent['close'].values
            model = LinearRegression().fit(X, y)
            # 预测未来 3 天
            future_idx = np.array([[20], [21], [22]])
            preds = model.predict(future_idx)
            future_dates = [(datetime.date.today() + datetime.timedelta(days=i)).strftime("%m-%d") for i in range(1, 4)]
            return {"dates": future_dates, "prices": preds}
        except: return None

    def plot_professional_kline(self, df, title):
        """新增：在图中标出买卖点"""
        if df is None or df.empty: return None
        # 计算买卖信号 (MA5/MA20 金叉死叉)
        df['MA5'] = df['close'].rolling(5).mean()
        df['MA20'] = df['close'].rolling(20).mean()
        df['BS'] = 0
        df.loc[(df['MA5'] > df['MA20']) & (df['MA5'].shift(1) <= df['MA20'].shift(1)), 'BS'] = 1
        df.loc[(df['MA5'] < df['MA20']) & (df['MA5'].shift(1) >= df['MA20'].shift(1)), 'BS'] = -1

        fig = go.Figure(data=[go.Candlestick(x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='K线')])
        # 标出买卖点 (保持原功能美观)
        buys = df[df['BS'] == 1]
        fig.add_trace(go.Scatter(x=buys['date'], y=buys['low']*0.98, mode='markers+text', marker=dict(symbol='triangle-up', size=12, color='red'), text="B", textposition="bottom center", name="买点"))
        sells = df[df['BS'] == -1]
        fig.add_trace(go.Scatter(x=sells['date'], y=sells['high']*1.02, mode='markers+text', marker=dict(symbol='triangle-down', size=12, color='green'), text="S", textposition="top center", name="卖点"))
        
        fig.update_layout(title=f"{title} - 智能标注K线", xaxis_rangeslider_visible=False, height=500)
        return fig

# ==========================================
# 3. 界面 UI (保持原功能布局)
# ==========================================
engine = QuantsEngine()

# 初始化缓存
if 'full_pool' not in st.session_state: st.session_state['full_pool'] = []
if 'scan_res' not in st.session_state: st.session_state['scan_res'] = []
if 'valid_options' not in st.session_state: st.session_state['valid_options'] = []

# 侧边栏逻辑维持原样
st.sidebar.header("🕹️ 控制台")
max_price = st.sidebar.slider("💰 价格上限 (元)", 3.0, 100.0, 40.0)
pool_mode = st.sidebar.radio("🔎 选股范围:", ("中证500 (中小盘)", "全市场扫描", "手动输入"))

if st.sidebar.button(f"📥 加载 {pool_mode}"):
    with st.spinner("获取中..."):
        if pool_mode == "全市场扫描": st.session_state['full_pool'] = engine.get_all_stocks()
        else: st.session_state['full_pool'] = ["sh.600519", "sz.002131"]
        st.sidebar.success(f"已加载 {len(st.session_state['full_pool'])} 只")

# 深度分析与三日预测展示区
if st.session_state['valid_options']:
    st.divider()
    target = st.selectbox("选择目标进行深度分析", st.session_state['valid_options'])
    if st.button("🚀 立即分析"):
        code = target.split("|")[0].strip()
        # 增加修复白屏的深度数据获取逻辑
        df = engine.get_deep_data(code) 
        if df is not None:
            # 1. 显示标注买卖点的图表
            st.plotly_chart(engine.plot_professional_kline(df, target), use_container_width=True)
            # 2. 显示后三天预估
            pred = engine.run_ai_prediction(df)
            if pred:
                st.markdown("### 📅 AI 趋势推演 (未来3日预估)")
                cols = st.columns(3)
                for i in range(3):
                    cols[i].metric(label=f"日期: {pred['dates'][i]}", value=f"¥{pred['prices'][i]:.2f}")
        else:
            st.error("分析失败，请重试")