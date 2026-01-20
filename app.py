import streamlit as st

# ==========================================
# ⚠️ 核心配置
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

STRATEGY_LOGIC = {
    "👑 四星共振": "近20日有涨停 + 向上跳空缺口 + 4连阳 + 量比>1.8",
    "🐲 妖股基因": "近60日涨停≥3次 + 获利筹码>80% + 上市>30天",
    "🔥 换手锁仓": "连续2日换手率>5% + 获利筹码>70%",
    "🔴 温和吸筹": "3连阳且累计涨幅<5% + 获利筹码>62%",
    "📈 多头排列": "昨日收阳 且 今日收盘价 > 昨日收盘价"
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
        """获取全市场股票，最多6000只"""
        try:
            bs.login()
            rs = bs.query_all_stock()
            stocks = []
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            
            for data in data_list:
                if len(data) >= 2:
                    code = data[0]
                    name = data[1] if len(data) > 1 else ""
                    if self.is_valid(code, name):
                        stocks.append(code)
            
            bs.logout()
            return stocks[:self.MAX_SCAN_LIMIT]
        except:
            try:
                bs.logout()
            except:
                pass
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

    def _process_single_stock(self, code, max_price=None):
        code = self.clean_code(code)
        end = datetime.datetime.now().strftime("%Y-%m-%d")
        start = (datetime.datetime.now() - datetime.timedelta(days=150)).strftime("%Y-%m-%d")
        
        data = []
        info = {'name': code, 'industry': '-', 'ipoDate': '2000-01-01'}
        
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
            if not self.is_valid(code, info['name']): return None
            rs = bs.query_history_k_data_plus(code, "date,open,close,high,low,volume,pctChg,turn", start_date=start, frequency="d", adjustflag="3")
            while rs.next(): data.append(rs.get_row_data())
        except: return None

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
        """扫描市场 - 保持原来的进度条逻辑"""
        results, alerts, valid_codes_list = [], [], []
        lg = bs.login()
        if lg.error_code != '0':
            st.error("连接服务器失败，请检查网络！")
            return [], [], []

        if len(code_list) > self.MAX_SCAN_LIMIT:
            code_list = code_list[:self.MAX_SCAN_LIMIT]
            st.info(f"⚠️ 股票数量超过限制，已截取前{self.MAX_SCAN_LIMIT}只")

        total = len(code_list)
        
        progress_container = st.empty()
        progress_bar = progress_container.progress(0, text=f"🚀 正在启动稳定扫描 (共 {total} 只)...")
        
        BATCH_SIZE = 20
        
        for i, code in enumerate(code_list):
            if i % BATCH_SIZE == 0 or i == total - 1:
                progress = (i + 1) / total
                current_count = min(i + 1, total)
                progress_bar.progress(progress, 
                                    text=f"🔍 正在分析: {code} ({current_count}/{total}) | 已命中: {len(results)} 只")
            
            try:
                res = self._process_single_stock(code, max_price)
                if res:
                    results.append(res["result"])
                    if res["alert"]: alerts.append(res["alert"])
                    valid_codes_list.append(res["option"])
            except:
                bs.logout()
                time.sleep(0.5)
                bs.login()
                continue

        bs.logout()
        progress_container.empty()
        return results, alerts, valid_codes_list

    def get_deep_data(self, code):
        """获取深度数据 - 修复白屏问题"""
        try:
            bs.login()
            # 缩短时间范围，避免数据过多
            end = datetime.datetime.now().strftime("%Y-%m-%d")
            start = (datetime.datetime.now() - datetime.timedelta(days=180)).strftime("%Y-%m-%d")
            
            # 只获取必要字段，避免复杂数据
            rs = bs.query_history_k_data_plus(
                code, 
                "date,open,close,high,low,volume",  # 移除peTTM,pbMRQ，可能为空
                start_date=start, 
                end_date=end, 
                frequency="d", 
                adjustflag="3"
            )
            
            if rs.error_code != '0':
                bs.logout()
                return None
                
            data = []
            while rs.next(): 
                data.append(rs.get_row_data())
            
            bs.logout()
            
            if not data: 
                return None
                
            df = pd.DataFrame(data, columns=["date", "open", "close", "high", "low", "volume"])
            
            # 安全转换数据类型
            for col in ["open", "close", "high", "low", "volume"]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 清理无效数据
            df = df.dropna(subset=['close', 'volume'])
            
            if len(df) < 20:  # 降低数据要求
                return None
                
            return df
            
        except Exception as e:
            try:
                bs.logout()
            except:
                pass
            return None

    def run_ai_prediction(self, df):
        """AI预测 - 增加异常处理"""
        if df is None or len(df) < 20:
            return None
            
        try:
            recent = df.tail(20).reset_index(drop=True)
            X = np.array(recent.index).reshape(-1, 1)
            y = recent['close'].values
            
            # 检查数据有效性
            if len(y) < 5 or np.isnan(y).any():
                return None
                
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
        except Exception as e:
            # 预测失败时返回简单信息
            return {
                "dates": ["明日", "后日", "大后日"],
                "prices": [0, 0, 0],
                "pred_price": 0,
                "title": "⚠️ 数据不足",
                "desc": "当前数据不足以进行准确预测",
                "action": "建议：补充数据后重试",
                "color": "blue"
            }

    def calc_indicators(self, df):
        """计算技术指标 - 增加异常处理"""
        if df is None or df.empty:
            return df
            
        try:
            df = df.copy()
            df['MA5'] = df['close'].rolling(5).mean()
            df['MA20'] = df['close'].rolling(20).mean()
            
            # 尝试计算MACD，但忽略错误
            try:
                exp1 = df['close'].ewm(span=12, adjust=False).mean()
                exp2 = df['close'].ewm(span=26, adjust=False).mean()
                df['DIF'] = exp1 - exp2
                df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
                df['MACD'] = 2 * (df['DIF'] - df['DEA'])
            except:
                pass
                
            return df
        except:
            return df

    def plot_professional_kline(self, df, title):
        """绘制K线图 - 增加异常处理"""
        if df is None or df.empty or len(df) < 10:
            return None
            
        try:
            df = self.calc_indicators(df)
            
            # 创建信号列，但安全处理
            df['Signal'] = 0
            if 'MA5' in df.columns and 'MA20' in df.columns:
                try:
                    df.loc[(df['MA5'] > df['MA20']) & (df['MA5'].shift(1) <= df['MA20'].shift(1)), 'Signal'] = 1 
                    df.loc[(df['MA5'] < df['MA20']) & (df['MA5'].shift(1) >= df['MA20'].shift(1)), 'Signal'] = -1 
                except:
                    pass

            buy_points = df[df['Signal'] == 1]
            sell_points = df[df['Signal'] == -1]

            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
                name='K线', increasing_line_color='red', decreasing_line_color='green'
            ))
            
            # 安全添加均线
            if 'MA5' in df.columns:
                fig.add_trace(go.Scatter(x=df['date'], y=df['MA5'], name='MA5', line=dict(color='orange', width=1)))
            
            if 'MA20' in df.columns:
                fig.add_trace(go.Scatter(x=df['date'], y=df['MA20'], name='MA20', line=dict(color='blue', width=1)))

            # 安全添加买卖点
            if not buy_points.empty:
                try:
                    fig.add_trace(go.Scatter(x=buy_points['date'], y=buy_points['low']*0.98, mode='markers+text', 
                                           marker=dict(symbol='triangle-up', size=12, color='red'), 
                                           text='B', textposition='bottom center', name='买入'))
                except:
                    pass
            
            if not sell_points.empty:
                try:
                    fig.add_trace(go.Scatter(x=sell_points['date'], y=sell_points['high']*1.02, mode='markers+text', 
                                           marker=dict(symbol='triangle-down', size=12, color='green'), 
                                           text='S', textposition='top center', name='卖出'))
                except:
                    pass

            fig.update_layout(title=f"{title} - 智能操盘K线", xaxis_rangeslider_visible=False, height=500)
            return fig
        except Exception as e:
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
if 'analyzing' not in st.session_state:
    st.session_state['analyzing'] = False

st.sidebar.header("🕹️ 控制台")
max_price_limit = st.sidebar.slider("💰 价格上限 (元)", 3.0, 100.0, 20.0)

pool_mode = st.sidebar.radio("🔎 选股范围:", ("中证500 (中小盘)", "沪深300 (大盘)", "全市场扫描", "手动输入"))

scan_limit = st.sidebar.slider("🔢 扫描数量 (池大小)", 50, 6000, 500, step=50)

if pool_mode == "手动输入":
    default_pool = "600519, 002131, 002312, 600580, 002594"
    target_pool_str = st.sidebar.text_area("监控股票池", default_pool, height=100)
    final_code_list = [code.strip() for code in target_pool_str.replace("，", ",").split(",") if code.strip()]
else:
    if st.sidebar.button(f"📥 加载 {pool_mode} 成分股"):
        with st.spinner("正在获取成分股..."):
            if pool_mode == "全市场扫描":
                stock_list = engine.get_all_stocks()
            elif "中证500" in pool_mode:
                index_code = "zz500"
                stock_list = engine.get_index_stocks(index_code)
            else:
                index_code = "hs300"
                stock_list = engine.get_index_stocks(index_code)
            
            if stock_list:
                st.session_state['full_pool'] = stock_list 
                st.sidebar.success(f"已加载全量 {len(stock_list)} 只股票")
            else:
                st.sidebar.error("获取股票失败，请重试")
    
    if 'full_pool' in st.session_state:
        full_list = st.session_state['full_pool']
        final_code_list = full_list[:scan_limit] 
        st.sidebar.info(f"池内待扫: {len(final_code_list)} 只 (总库: {len(full_list)})")
    else:
        final_code_list = []

st.sidebar.markdown("---")
if st.sidebar.button("🚀 启动全策略扫描 (V45)", type="primary"):
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
        alert_names = "、".join(alerts[:5])  # 限制显示数量
        st.success(f"🔥 发现 {len(alerts)} 只【主力高控盘】标的：**{alert_names}**")
    
    df_scan = pd.DataFrame(results).sort_values(by="priority", ascending=False)
    
    if df_scan.empty:
        st.warning(f"⚠️ 扫描完成，无符合条件的股票。")
    else:
        if len(df_scan) > 100:
            page_size = 50
            total_pages = max(1, (len(df_scan) + page_size - 1) // page_size)
            
            page_num = st.number_input("📄 页码", min_value=1, max_value=total_pages, value=1)
            start_idx = (page_num - 1) * page_size
            end_idx = min(start_idx + page_size, len(df_scan))
            display_df = df_scan.iloc[start_idx:end_idx]
            
            st.caption(f"显示第 {start_idx+1}-{end_idx} 条，共 {len(df_scan)} 条 (第 {page_num}/{total_pages} 页)")
        else:
            display_df = df_scan
        
        st.dataframe(
            display_df, 
            hide_index=True,
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
    st.info("👈 请在左侧加载股票 -> 点击'启动全策略扫描'")

st.divider()

if 'valid_options' in st.session_state and st.session_state['valid_options']:
    st.subheader("🧠 深度分析")
    target = st.selectbox("选择目标进行深度分析", st.session_state['valid_options'])
    
    target_code = target.split("|")[0].strip()
    target_name = target.split("|")[1].strip()

    if st.button(f"🚀 立即分析 {target_name}", key="analyze_btn"):
        # 设置分析状态
        st.session_state['analyzing'] = True
        
        # 使用try-except包装整个分析过程
        try:
            with st.spinner("AI 正在推演未来变盘点..."):
                # 获取数据 - 添加更多错误处理
                df = engine.get_deep_data(target_code)
                
                if df is not None and not df.empty:
                    # 基本信息
                    last = df.iloc[-1]
                    col1, col2, col3 = st.columns(3)
                    col1.metric("当前价格", f"¥{last['close']:.2f}")
                    
                    # AI预测
                    future_info = engine.run_ai_prediction(df)
                    
                    if future_info and future_info['pred_price'] > 0:
                        col2.metric("AI预测明日", f"¥{future_info['pred_price']:.2f}", 
                                   delta=f"{future_info['pred_price']-last['close']:.2f}", 
                                   delta_color="inverse")
                        
                        if future_info['color'] == 'red':
                            st.error(f"### {future_info['title']}\n{future_info['desc']}\n\n**{future_info['action']}**")
                        elif future_info['color'] == 'green':
                            st.success(f"### {future_info['title']}\n{future_info['desc']}\n\n**{future_info['action']}**")
                        else:
                            st.info(f"### {future_info['title']}\n{future_info['desc']}\n\n**{future_info['action']}**")

                        st.markdown("### 📅 AI 时空推演 (未来3日)")
                        d_cols = st.columns(3)
                        for i in range(3):
                            d_cols[i].metric(label=future_info['dates'][i], 
                                           value=f"¥{future_info['prices'][i]:.2f}", 
                                           delta="预测")
                    else:
                        col2.metric("AI预测明日", f"¥{last['close']:.2f}", delta="数据不足")
                        st.warning("⚠️ 数据不足以进行AI预测，显示当前价格")
                    
                    col3.metric("数据天数", len(df))
                    
                    # K线图
                    st.markdown("### 📊 K线分析")
                    fig = engine.plot_professional_kline(df, target_name)
                    
                    if fig:
                        st.plotly_chart(fig, width='stretch')
                        st.info("💡 **图例**: 🔺红色B=金叉买点 | 🔻绿色S=死叉卖点 (仅供辅助参考)")
                    else:
                        st.warning("⚠️ 无法生成K线图，数据可能不足")
                        
                    # 显示最近数据
                    with st.expander("📋 查看最近交易数据"):
                        st.dataframe(df.tail(10))
                        
                else:
                    st.error("❌ 无法获取该股票的详细数据，请尝试重新扫描或选择其他股票")
                    
        except Exception as e:
            st.error(f"❌ 分析过程中出现错误: {str(e)[:100]}")
            st.info("💡 建议：请重试或选择其他股票进行分析")
            
        finally:
            # 重置分析状态
            st.session_state['analyzing'] = False

# 添加系统状态信息
with st.expander("📊 系统状态", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        if 'full_pool' in st.session_state:
            st.metric("股票池总量", f"{len(st.session_state['full_pool']):,}")
        else:
            st.metric("股票池总量", "0")
    
    with col2:
        if 'scan_res' in st.session_state:
            st.metric("当前结果数", f"{len(st.session_state['scan_res']):,}")
        else:
            st.metric("当前结果数", "0")
    
    if 'valid_options' in st.session_state:
        st.write(f"可选分析股票: {len(st.session_state['valid_options'])} 只")
    
    st.write(f"最大扫描限制: {engine.MAX_SCAN_LIMIT:,} 只")
    st.write(f"当前时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 添加使用提示
st.caption("""
💡 **使用提示**: 
1. 扫描大量股票时请耐心等待，进度条会正常显示扫描进度
2. 点击"分析"按钮时，系统会安全获取数据，避免白屏
3. 如果某只股票分析失败，请尝试选择其他股票
4. 投资有风险，决策需谨慎
""")