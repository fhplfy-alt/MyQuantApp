import streamlit as st

# ==========================================
# ⚠️ 核心配置
# ==========================================
st.set_page_config(
    page_title="V45 超级扫描版", 
    layout="wide", 
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

st.title("🛡️ V45 智能量化系统 (6000股扫描版)")
st.caption("✅ 系统已就绪 | 支持6000股批量扫描 | V45 Build")

# ==========================================
# 1. 安全导入
# ==========================================
try:
    import plotly.graph_objects as go
    import baostock as bs
    import pandas as pd
    import numpy as np
    import time
    import datetime
    import concurrent.futures
    from sklearn.linear_model import LinearRegression
except ImportError as e:
    st.error(f"❌ 启动失败！缺少必要运行库: {e}")
    st.stop()

# ==========================================
# 0. 全局配置
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

# ==========================================
# 2. 核心引擎 - 支持6000股扫描
# ==========================================
class QuantsEngine:
    def __init__(self):
        self.batch_size = 50  # 每批次处理数量
        self.max_workers = 4  # 最大并行线程数
        self.scan_limit = 6000  # 最大扫描数量
        
    def clean_code(self, code):
        code = str(code).strip()
        if not (code.startswith('sh.') or code.startswith('sz.')):
            return f"sh.{code}" if code.startswith('6') else f"sz.{code}"
        return code

    def is_valid(self, code, name):
        """验证股票是否有效"""
        if "sh.688" in code: 
            return False  # 科创板
        if "bj." in code or code.startswith("sz.8") or code.startswith("sz.4"): 
            return False  # 北交所和退市板块
        if "ST" in name or "*" in name or "退" in name: 
            return False  # ST股票和退市股
        return True

    def get_all_stocks(self):
        """获取全市场股票列表（支持6000只）"""
        try:
            lg = bs.login()
            if lg.error_code != '0':
                st.error(f"登录baostock失败: {lg.error_msg}")
                return []
            
            # 获取所有股票
            rs = bs.query_all_stock()
            if rs.error_code != '0':
                st.error(f"获取股票列表失败: {rs.error_msg}")
                return []
            
            all_stocks = []
            while (rs.error_code == '0') & rs.next():
                row = rs.get_row_data()
                if len(row) >= 2:
                    code = row[0]
                    name = row[1]
                    # 过滤无效股票
                    if self.is_valid(code, name):
                        all_stocks.append(code)
            
            bs.logout()
            
            # 去重并限制数量
            unique_stocks = list(dict.fromkeys(all_stocks))
            return unique_stocks[:self.scan_limit]
            
        except Exception as e:
            st.error(f"获取股票列表时出错: {str(e)}")
            try:
                bs.logout()
            except:
                pass
            return []

    def get_index_stocks(self, index_type="all"):
        """获取指数成分股或全市场股票"""
        try:
            lg = bs.login()
            if lg.error_code != '0':
                st.error(f"登录baostock失败: {lg.error_msg}")
                return []
            
            stocks = []
            
            if index_type == "all":
                # 获取全市场股票
                rs = bs.query_all_stock()
                while (rs.error_code == '0') & rs.next():
                    row = rs.get_row_data()
                    if len(row) >= 2:
                        code = row[0]
                        name = row[1]
                        if self.is_valid(code, name):
                            stocks.append(code)
                
            elif index_type == "hs300":
                rs = bs.query_hs300_stocks()
                while (rs.error_code == '0') & rs.next():
                    row = rs.get_row_data()
                    if len(row) > 1:
                        stocks.append(row[1])
                        
            elif index_type == "zz500":
                rs = bs.query_zz500_stocks()
                while (rs.error_code == '0') & rs.next():
                    row = rs.get_row_data()
                    if len(row) > 1:
                        stocks.append(row[1])
            
            bs.logout()
            
            # 去重并限制数量
            unique_stocks = list(dict.fromkeys(stocks))
            return unique_stocks[:self.scan_limit]
            
        except Exception as e:
            st.error(f"获取股票列表时出错: {str(e)}")
            try:
                bs.logout()
            except:
                pass
            return []

    def process_batch(self, batch_codes, max_price=None):
        """处理一个批次的股票"""
        batch_results = []
        batch_alerts = []
        batch_options = []
        
        for code in batch_codes:
            try:
                res = self._process_single_stock(code, max_price)
                if res:
                    batch_results.append(res["result"])
                    if res["alert"]: 
                        batch_alerts.append(res["alert"])
                    batch_options.append(res["option"])
            except Exception as e:
                continue
                
        return batch_results, batch_alerts, batch_options

    def calc_winner_rate(self, df, current_price):
        """计算获利筹码比例"""
        if df.empty or current_price <= 0:
            return 50.0
            
        # 使用近期数据计算
        recent_df = df.tail(60) if len(df) >= 60 else df
        
        if len(recent_df) < 10:
            return 50.0
            
        low_price = recent_df['low'].min()
        high_price = recent_df['high'].max()
        
        if high_price <= low_price:
            return 50.0
            
        # 计算价格位置
        position = (current_price - low_price) / (high_price - low_price)
        winner_rate = max(20.0, min(95.0, (1 - position) * 100))
        
        return winner_rate

    def _process_single_stock(self, code, max_price=None):
        """处理单个股票"""
        code = self.clean_code(code)
        
        # 设置时间范围
        end_date = datetime.datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.datetime.now() - datetime.timedelta(days=120)).strftime("%Y-%m-%d")
        
        data = []
        info = {'name': code, 'industry': '-', 'ipoDate': '2000-01-01'}
        
        try:
            # 登录获取数据
            lg = bs.login()
            if lg.error_code != '0':
                return None
            
            # 获取基本信息
            rs_info = bs.query_stock_basic(code=code)
            if rs_info.error_code != '0': 
                bs.logout()
                return None
                
            if rs_info.next():
                row = rs_info.get_row_data()
                info['name'] = row[1] if row[1] else code
                info['ipoDate'] = row[2] if row[2] else '2000-01-01'
            
            # 检查有效性
            if not self.is_valid(code, info['name']): 
                bs.logout()
                return None
            
            # 获取K线数据
            rs = bs.query_history_k_data_plus(
                code, 
                "date,open,close,high,low,volume,pctChg,turn", 
                start_date=start_date, 
                end_date=end_date,
                frequency="d", 
                adjustflag="3"
            )
            
            if rs.error_code != '0':
                bs.logout()
                return None
            
            # 获取数据
            while (rs.error_code == '0') & rs.next():
                row_data = rs.get_row_data()
                if len(row_data) == 8:
                    data.append(row_data)
            
            bs.logout()
                
        except Exception as e:
            try:
                bs.logout()
            except:
                pass
            return None

        if not data or len(data) < 30:
            return None
            
        try:
            # 创建DataFrame
            df = pd.DataFrame(data, columns=["date", "open", "close", "high", "low", "volume", "pctChg", "turn"])
            
            # 转换数据类型
            numeric_cols = ["open", "close", "high", "low", "volume", "pctChg", "turn"]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 清理数据
            df = df.dropna(subset=["close", "volume"])
            if len(df) < 30:
                return None
                
        except Exception as e:
            return None
        
        # 获取最新数据
        curr = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else curr
        
        # 价格过滤
        if max_price is not None and curr['close'] > max_price:
            return None
        
        # 计算指标
        winner_rate = self.calc_winner_rate(df, curr['close'])
        
        # 计算移动平均线
        df['MA5'] = df['close'].rolling(window=5, min_periods=1).mean()
        df['MA20'] = df['close'].rolling(window=20, min_periods=1).mean()
        df['MA60'] = df['close'].rolling(window=60, min_periods=1).mean()
        
        # 风险评估
        risk_level = "未知"
        if not pd.isna(df['MA5'].iloc[-1]):
            bias = (curr['close'] - df['MA5'].iloc[-1]) / df['MA5'].iloc[-1] * 100
            if bias > 15: 
                risk_level = "High"
            elif bias < -10: 
                risk_level = "Med"
            elif curr['close'] < df['MA20'].iloc[-1]: 
                risk_level = "Med"
            else: 
                risk_level = "Low"
        
        # 策略信号检测
        signal_tags = []
        priority = 0
        action = "WAIT"
        
        # 策略1: 价格上涨
        if curr['pctChg'] > 0:
            signal_tags.append("📈上涨")
            priority = 10
            action = "HOLD"
        
        # 策略2: 多头排列
        if len(df) >= 5:
            ma5 = df['MA5'].iloc[-1]
            ma20 = df['MA20'].iloc[-1]
            if not pd.isna(ma5) and not pd.isna(ma20):
                if curr['close'] > ma5 and ma5 > ma20:
                    signal_tags.append("📈多头")
                    priority = max(priority, 20)
                    action = "BUY"
        
        # 策略3: 量价齐升
        avg_volume = df['volume'].mean()
        if curr['volume'] > avg_volume * 1.2 and curr['pctChg'] > 1:
            signal_tags.append("📊量价升")
            priority = max(priority, 30)
            action = "BUY"
        
        # 策略4: 温和上涨
        if len(df) >= 3:
            recent_3 = df.tail(3)
            if all(recent_3['pctChg'] > 0) and recent_3['pctChg'].sum() < 10:
                signal_tags.append("🔴温和涨")
                priority = max(priority, 40)
                action = "BUY"
        
        # 策略5: 金叉信号
        if len(df) >= 20:
            curr_ma5 = df['MA5'].iloc[-1]
            curr_ma20 = df['MA20'].iloc[-1]
            prev_ma5 = df['MA5'].iloc[-2]
            prev_ma20 = df['MA20'].iloc[-2]
            
            if not pd.isna(curr_ma5) and not pd.isna(curr_ma20):
                if prev_ma5 <= prev_ma20 and curr_ma5 > curr_ma20:
                    signal_tags.append("🚀金叉")
                    priority = max(priority, 50)
                    action = "BUY"
        
        # 如果没有任何信号，返回None
        if priority == 0:
            return None
        
        # 返回结果
        return {
            "result": {
                "代码": code,
                "名称": info['name'][:8],
                "现价": f"{curr['close']:.2f}",
                "涨跌": f"{curr['pctChg']:.2f}%",
                "获利筹码": f"{winner_rate:.1f}%",
                "风险评级": risk_level,
                "策略信号": " ".join(signal_tags),
                "综合评级": action,
                "priority": priority
            },
            "alert": f"{info['name'][:8]}" if priority >= 40 else None,
            "option": f"{code} | {info['name'][:8]}"
        }

    def scan_massive_stocks(self, code_list, max_price=None, batch_size=50):
        """大规模扫描股票（支持6000只）"""
        if not code_list:
            return [], [], []
        
        total = len(code_list)
        if total == 0:
            return [], [], []
        
        # 分批次处理
        batches = [code_list[i:i + batch_size] for i in range(0, total, batch_size)]
        total_batches = len(batches)
        
        # 进度显示
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_results = []
        all_alerts = []
        all_options = []
        
        # 处理每个批次
        for batch_idx, batch_codes in enumerate(batches):
            # 更新进度
            progress = (batch_idx + 1) / total_batches
            progress_bar.progress(progress)
            
            current_count = batch_idx * batch_size + len(batch_codes)
            status_text.text(f"批次 {batch_idx+1}/{total_batches} | 处理中: {current_count}/{total}")
            
            # 处理当前批次
            batch_results, batch_alerts, batch_options = self.process_batch(batch_codes, max_price)
            
            # 收集结果
            all_results.extend(batch_results)
            all_alerts.extend(batch_alerts)
            all_options.extend(batch_options)
            
            # 小延迟避免请求过快
            time.sleep(0.5)
        
        # 清理进度显示
        progress_bar.empty()
        status_text.empty()
        
        return all_results, all_alerts, all_options

    def get_deep_data(self, code):
        """获取深度数据"""
        try:
            lg = bs.login()
            if lg.error_code != '0':
                return None
            
            end_date = datetime.datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.datetime.now() - datetime.timedelta(days=90)).strftime("%Y-%m-%d")
            
            rs = bs.query_history_k_data_plus(
                code, 
                "date,open,close,high,low,volume,pctChg", 
                start_date=start_date, 
                end_date=end_date, 
                frequency="d", 
                adjustflag="3"
            )
            
            if rs.error_code != '0':
                bs.logout()
                return None
            
            data = []
            while (rs.error_code == '0') & rs.next():
                row_data = rs.get_row_data()
                if len(row_data) == 7:
                    data.append(row_data)
            
            bs.logout()
            
            if not data:
                return None
                
            df = pd.DataFrame(data, columns=["date", "open", "close", "high", "low", "volume", "pctChg"])
            
            numeric_cols = ["open", "close", "high", "low", "volume", "pctChg"]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna(subset=["close", "volume"])
            
            if len(df) < 10:
                return None
                
            return df
            
        except Exception as e:
            try:
                bs.logout()
            except:
                pass
            return None

    def run_ai_prediction(self, df):
        """AI预测"""
        if df is None or len(df) < 20:
            return None
            
        try:
            recent = df.tail(20).reset_index(drop=True)
            X = np.array(recent.index).reshape(-1, 1)
            y = recent['close'].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            last_idx = recent.index[-1]
            future_idx = np.array([[last_idx + 1]])
            pred_price = model.predict(future_idx)[0]
            
            slope = model.coef_[0]
            
            if slope > 0.01:
                hint_title = "📈 上升趋势"
                hint_desc = f"趋势向上，预计明日 ¥{pred_price:.2f}"
                action = "持有或逢低买入"
                color = "green"
            elif slope > -0.01:
                hint_title = "⚖️ 横盘震荡"
                hint_desc = f"震荡整理，预计明日 ¥{pred_price:.2f}"
                action = "观望等待"
                color = "blue"
            else:
                hint_title = "📉 下跌趋势"
                hint_desc = f"趋势向下，建议谨慎"
                action = "控制风险"
                color = "orange"

            return {
                "pred_price": pred_price,
                "title": hint_title,
                "desc": hint_desc,
                "action": action,
                "color": color
            }
        except:
            return None

    def plot_kline(self, df, title):
        """绘制K线图"""
        if df is None or df.empty or len(df) < 10:
            return None
            
        try:
            df = df.copy()
            df['MA5'] = df['close'].rolling(5, min_periods=1).mean()
            df['MA10'] = df['close'].rolling(10, min_periods=1).mean()
            
            fig = go.Figure()
            
            fig.add_trace(go.Candlestick(
                x=df['date'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='K线',
                increasing_line_color='red',
                decreasing_line_color='green'
            ))
            
            if 'MA5' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['date'],
                    y=df['MA5'],
                    name='MA5',
                    line=dict(color='orange', width=1)
                ))
            
            fig.update_layout(
                title=f"{title} - K线图",
                xaxis_title="日期",
                yaxis_title="价格",
                xaxis_rangeslider_visible=False,
                height=400
            )
            
            return fig
        except:
            return None

# ==========================================
# 3. 界面 UI
# ==========================================
engine = QuantsEngine()

# 初始化session_state
if 'full_pool' not in st.session_state:
    st.session_state['full_pool'] = []
if 'scan_res' not in st.session_state:
    st.session_state['scan_res'] = []
if 'valid_options' not in st.session_state:
    st.session_state['valid_options'] = []
if 'alerts' not in st.session_state:
    st.session_state['alerts'] = []
if 'scan_complete' not in st.session_state:
    st.session_state['scan_complete'] = False

st.sidebar.header("🕹️ 控制台")

# 扫描模式选择
scan_mode = st.sidebar.selectbox(
    "🔎 扫描模式",
    ["全市场扫描 (6000股)", "沪深300", "中证500", "手动输入", "快速测试"]
)

# 价格上限
max_price_limit = st.sidebar.slider("💰 价格上限 (元)", 1.0, 200.0, 100.0, 1.0)

# 扫描数量控制
if scan_mode == "全市场扫描 (6000股)":
    max_scan = 6000
    default_scan = 1000
elif scan_mode == "沪深300":
    max_scan = 600
    default_scan = 300
elif scan_mode == "中证500":
    max_scan = 800
    default_scan = 500
else:
    max_scan = 200
    default_scan = 50

scan_limit = st.sidebar.slider(f"🔢 扫描数量 (最大{max_scan})", 10, max_scan, min(default_scan, max_scan), 10)

st.sidebar.markdown("---")

# 股票池管理
if scan_mode == "手动输入":
    default_pool = """600519,000858,000333,002415,300750,600036,601318"""
    target_pool_str = st.sidebar.text_area("📝 输入股票代码", default_pool, height=100)
    final_code_list = [code.strip() for code in target_pool_str.replace("，", ",").split(",") if code.strip()]
    
    if final_code_list:
        st.sidebar.success(f"✅ 已加载 {len(final_code_list)} 只股票")
    
elif scan_mode == "快速测试":
    test_codes = [
        "600519", "000858", "000333", "002415", "300750",
        "600036", "601318", "000001", "600030", "000002"
    ]
    final_code_list = test_codes[:scan_limit]
    st.sidebar.info(f"🧪 测试模式: {len(final_code_list)} 只股票")
    
else:
    # 获取股票列表
    if st.sidebar.button(f"📥 加载{scan_mode}股票", type="primary"):
        with st.spinner(f"正在获取{scan_mode}股票列表..."):
            if scan_mode == "全市场扫描 (6000股)":
                stock_list = engine.get_all_stocks()
            elif scan_mode == "沪深300":
                stock_list = engine.get_index_stocks("hs300")
            elif scan_mode == "中证500":
                stock_list = engine.get_index_stocks("zz500")
            else:
                stock_list = []
            
            if stock_list:
                st.session_state['full_pool'] = stock_list
                st.sidebar.success(f"✅ 已加载 {len(stock_list)} 只股票")
            else:
                st.sidebar.error("❌ 获取股票列表失败")
    
    if st.session_state['full_pool']:
        full_list = st.session_state['full_pool']
        final_code_list = full_list[:scan_limit]
        st.sidebar.info(f"📊 池内待扫: {len(final_code_list)} 只")
    else:
        final_code_list = []

# 扫描控制
st.sidebar.markdown("---")

# 批量大小设置
batch_size = st.sidebar.selectbox("🔧 批量大小", [20, 50, 100, 200], index=1)

# 扫描按钮
if st.sidebar.button("🚀 启动大规模扫描", type="primary", use_container_width=True):
    if not final_code_list:
        st.sidebar.error("❌ 请先加载股票池！")
    else:
        # 清空之前的结果
        st.session_state['scan_res'] = []
        st.session_state['valid_options'] = []
        st.session_state['alerts'] = []
        st.session_state['scan_complete'] = False
        
        # 显示扫描信息
        st.info(f"🔍 开始扫描: {len(final_code_list)} 只股票 | 价格 < {max_price_limit}元")
        
        # 执行扫描
        scan_res, alerts, valid_options = engine.scan_massive_stocks(
            final_code_list, 
            max_price=max_price_limit,
            batch_size=batch_size
        )
        
        # 保存结果
        st.session_state['scan_res'] = scan_res
        st.session_state['valid_options'] = valid_options
        st.session_state['alerts'] = alerts
        st.session_state['scan_complete'] = True
        
        # 显示扫描结果
        if scan_res:
            st.success(f"✅ 扫描完成！发现 {len(scan_res)} 只符合条件的股票")
        else:
            st.warning("⚠️ 扫描完成，但未发现符合条件的股票")

# 策略说明
with st.expander("📖 策略说明", expanded=True):
    st.markdown("### 🎯 大规模扫描策略")
    st.markdown("""
    本系统支持扫描**6000只股票**，采用分批次处理：
    
    **核心策略**：
    1. 📈 **上涨趋势** - 当日涨幅为正
    2. 📈 **多头排列** - 均线多头排列
    3. 📊 **量价齐升** - 成交量放大且价格上涨
    4. 🔴 **温和上涨** - 连续小幅上涨
    5. 🚀 **金叉信号** - 技术指标金叉
    
    **扫描特点**：
    - 🔧 支持6000股大规模扫描
    - 📊 分批次处理，避免内存溢出
    - ⚡ 实时进度显示
    - 💾 智能结果筛选
    
    **⚠️ 注意**：扫描结果仅供参考，投资需谨慎！
    """)

# 显示扫描结果
st.subheader(f"⚡ 扫描结果 (价格 < {max_price_limit}元)")

if st.session_state['scan_complete'] and st.session_state['scan_res']:
    results = st.session_state['scan_res']
    alerts = st.session_state['alerts']
    
    # 显示扫描统计
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📈 扫描总数", len(final_code_list))
    with col2:
        st.metric("✅ 命中数量", len(results))
    with col3:
        hit_rate = (len(results) / len(final_code_list) * 100) if final_code_list else 0
        st.metric("🎯 命中率", f"{hit_rate:.1f}%")
    with col4:
        st.metric("🔥 高潜力", len(alerts))
    
    # 显示高潜力标的
    if alerts:
        st.success(f"🎯 **高潜力标的**: {len(alerts)} 只")
        # 只显示前5个
        if len(alerts) > 5:
            st.info(f"高潜力股票: {', '.join(alerts[:5])} 等{len(alerts)}只")
        else:
            st.info(f"高潜力股票: {', '.join(alerts)}")
    
    # 转换为DataFrame
    if results:
        df_scan = pd.DataFrame(results)
        
        if not df_scan.empty:
            # 按优先级排序
            if 'priority' in df_scan.columns:
                df_scan = df_scan.sort_values(by="priority", ascending=False)
            
            # 分页显示结果
            page_size = 20
            total_pages = max(1, (len(df_scan) + page_size - 1) // page_size)
            
            if total_pages > 1:
                page_num = st.number_input("📄 页码", min_value=1, max_value=total_pages, value=1)
                start_idx = (page_num - 1) * page_size
                end_idx = min(start_idx + page_size, len(df_scan))
                display_df = df_scan.iloc[start_idx:end_idx]
                
                st.caption(f"显示第 {start_idx+1}-{end_idx} 条，共 {len(df_scan)} 条 (第 {page_num}/{total_pages} 页)")
            else:
                display_df = df_scan
            
            # 显示数据
            st.dataframe(
                display_df,
                hide_index=True,
                column_config={
                    "代码": st.column_config.TextColumn("代码", width="small"),
                    "名称": st.column_config.TextColumn("名称", width="small"),
                    "现价": st.column_config.NumberColumn("现价", format="%.2f", width="small"),
                    "涨跌": st.column_config.TextColumn("涨跌", width="small"),
                    "获利筹码": st.column_config.TextColumn("筹码%", width="small"),
                    "风险评级": st.column_config.TextColumn("风险", width="small"),
                    "策略信号": st.column_config.TextColumn("信号", width="medium", help=STRATEGY_TIP),
                    "综合评级": st.column_config.TextColumn("操作", width="small", help=ACTION_TIP),
                    "priority": None
                }
            )
            
            # 提供下载功能
            if not df_scan.empty:
                csv_data = df_scan.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 下载扫描结果 (CSV)",
                    data=csv_data,
                    file_name=f"scan_result_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        else:
            st.warning("⚠️ 暂无符合条件的股票")
    else:
        st.warning("⚠️ 暂无扫描结果")
else:
    st.info("👈 请在左侧配置参数并启动扫描")

# 深度分析部分
st.divider()
st.subheader("🧠 深度分析")

if st.session_state['valid_options']:
    target_options = st.session_state['valid_options']
    
    # 简化显示选项
    display_options = [opt[:50] for opt in target_options]  # 限制显示长度
    
    if display_options:
        target = st.selectbox(
            "选择股票进行深度分析", 
            display_options,
            help="选择扫描结果中的股票进行详细分析",
            index=0 if display_options else None
        )
        
        if target:
            # 找到完整的选项
            original_option = next((opt for opt in target_options if opt.startswith(target.split('...')[0])), target_options[0])
            
            try:
                target_code = original_option.split("|")[0].strip()
                target_name = original_option.split("|")[1].strip()
            except:
                target_code = original_option
                target_name = original_option
            
            if st.button(f"🔍 分析 {target_name[:15]}", type="primary"):
                try:
                    with st.spinner("正在分析中..."):
                        # 获取数据
                        df = engine.get_deep_data(target_code)
                        
                        if df is not None and not df.empty:
                            # 显示基本信息
                            st.markdown(f"### 📊 {target_name} ({target_code})")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            latest = df.iloc[-1]
                            prev = df.iloc[-2] if len(df) > 1 else latest
                            
                            with col1:
                                price_change = latest['close'] - prev['close']
                                st.metric("当前价格", f"¥{latest['close']:.2f}", delta=f"{price_change:.2f}")
                            with col2:
                                st.metric("今日涨跌", f"{latest['pctChg']:.2f}%")
                            with col3:
                                st.metric("最高价", f"¥{latest['high']:.2f}")
                            with col4:
                                st.metric("最低价", f"¥{latest['low']:.2f}")
                            
                            # AI预测
                            st.markdown("### 🤖 AI趋势预测")
                            future_info = engine.run_ai_prediction(df)
                            
                            if future_info:
                                if future_info['color'] == 'green':
                                    st.success(f"**{future_info['title']}**")
                                elif future_info['color'] == 'orange':
                                    st.warning(f"**{future_info['title']}**")
                                else:
                                    st.info(f"**{future_info['title']}**")
                                
                                st.write(future_info['desc'])
                                st.write(f"**操作建议:** {future_info['action']}")
                                st.write(f"**预测明日价格:** ¥{future_info['pred_price']:.2f}")
                            else:
                                st.info("数据不足进行AI预测")
                            
                            # K线图
                            st.markdown("### 📈 K线分析")
                            fig = engine.plot_kline(df, target_name)
                            
                            if fig:
                                st.plotly_chart(fig, width='stretch')
                                st.caption("💡 提示: 橙色线为5日均线")
                            else:
                                st.warning("无法生成K线图")
                                
                        else:
                            st.error("无法获取该股票的详细数据")
                except Exception as e:
                    st.error(f"分析过程中出错: {str(e)}")
else:
    st.info("👆 请先完成扫描以选择分析目标")

# 底部说明
st.divider()
st.caption(f"""
💡 **大规模扫描系统使用提示**: 
1. **全市场扫描**模式支持扫描6000只股票
2. 建议使用**50-100的批量大小**以获得最佳性能
3. 扫描过程中请勿关闭页面
4. 结果支持**分页查看和下载**
5. **⚠️ 重要**: 扫描大量股票需要较长时间，请耐心等待
6. 投资有风险，决策需谨慎
""")

# 性能统计
with st.expander("📊 系统状态", expanded=False):
    if 'full_pool' in st.session_state:
        st.write(f"股票池大小: {len(st.session_state['full_pool'])}")
    if 'scan_res' in st.session_state:
        st.write(f"扫描结果: {len(st.session_state['scan_res'])} 条")
    if 'valid_options' in st.session_state:
        st.write(f"可选分析: {len(st.session_state['valid_options'])} 只")
    
    st.write(f"当前时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.write(f"引擎配置: 批量大小={batch_size}, 最大扫描={engine.scan_limit}")