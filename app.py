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
st.caption("✅ 系统已就绪 | 核心组件加载完成 | V45 Build")

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
# 0. 全局配置 (🔥 核心修改区：说明书补全 🔥)
# ==========================================
# 这里补全了你在表格里可能看到的所有信号
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
# 2. 核心引擎 (V44 稳定内核保持不变)
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
            if index_type == "hs300": 
                rs = bs.query_hs300_stocks()
            else: 
                rs = bs.query_zz500_stocks()
            while rs.next(): 
                stocks.append(rs.get_row_data()[1])
        except Exception as e:
            st.warning(f"获取指数成分股时出错: {e}")
        finally: 
            bs.logout()
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
        
        try:
            # 先登录获取数据
            bs.login()
            
            rs_info = bs.query_stock_basic(code=code)
            if rs_info.error_code != '0': 
                bs.logout()
                return None
                
            if rs_info.next():
                row = rs_info.get_row_data()
                info['name'] = row[1] if row[1] else code
                info['ipoDate'] = row[2] if row[2] else '2000-01-01'
                
            rs_ind = bs.query_stock_industry(code)
            if rs_ind.next():
                info['industry'] = rs_ind.get_row_data()[3] if rs_ind.get_row_data()[3] else '-'
                
            if not self.is_valid(code, info['name']): 
                bs.logout()
                return None
                
            rs = bs.query_history_k_data_plus(
                code, 
                "date,open,close,high,low,volume,pctChg,turn", 
                start_date=start, 
                end_date=end,
                frequency="d", 
                adjustflag="3"
            )
            
            while rs.next(): 
                data.append(rs.get_row_data())
                
            bs.logout()
                
        except Exception as e:
            try:
                bs.logout()
            except:
                pass
            return None

        if not data: 
            return None
            
        try:
            df = pd.DataFrame(data, columns=["date", "open", "close", "high", "low", "volume", "pctChg", "turn"])
            # 转换数据类型
            for col in ["open", "close", "high", "low", "volume", "pctChg", "turn"]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception as e:
            return None
            
        if len(df) < 60: 
            return None

        # 确保有足够的数据
        df = df.dropna()
        if len(df) < 60:
            return None
            
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        if max_price is not None:
            if curr['close'] > max_price: 
                return None

        winner_rate = self.calc_winner_rate(df, curr['close'])
        
        try: 
            ipo_date = datetime.datetime.strptime(info['ipoDate'], "%Y-%m-%d")
        except: 
            ipo_date = datetime.datetime(2000, 1, 1)
            
        days_listed = (datetime.datetime.now() - ipo_date).days

        df['MA5'] = df['close'].rolling(5).mean()
        df['MA20'] = df['close'].rolling(20).mean()
        
        # 确保移动平均线有值
        if pd.isna(df['MA5'].iloc[-1]) or pd.isna(df['MA20'].iloc[-1]):
            risk_level = "未知"
        else:
            risk_level = self.calc_risk_level(curr['close'], df['MA5'].iloc[-1], df['MA20'].iloc[-1])

        signal_tags = []
        priority = 0
        action = "WAIT (观望)"

        # 检查3连阳
        is_3_up = False
        if len(df) >= 3:
            is_3_up = all(df['pctChg'].tail(3) > 0)
            sum_3_rise = df['pctChg'].tail(3).sum()
            
        if (is_3_up and sum_3_rise <= 5 and winner_rate > 62):
            signal_tags.append("🔴温和吸筹")
            priority = max(priority, 60)
            action = "BUY (低吸)"

        # 检查高换手
        is_high_turn = False
        if len(df) >= 2:
            is_high_turn = all(df['turn'].tail(2) > 5)
            
        if is_high_turn and winner_rate > 70:
            signal_tags.append("🔥换手锁仓")
            priority = max(priority, 70)
            action = "BUY (博弈)"

        # 检查妖股基因
        df_60 = df.tail(60)
        limit_up_60 = len(df_60[df_60['pctChg'] > 9.5])
        
        if limit_up_60 >= 3 and winner_rate > 80 and days_listed > 30:
            signal_tags.append("🐲妖股基因")
            priority = max(priority, 90)
            action = "STRONG BUY"

        # 检查四星共振
        recent_20 = df.tail(20)
        has_limit_up_20 = len(recent_20[recent_20['pctChg'] > 9.5]) > 0
        
        has_gap = False
        recent_10 = df.tail(10).reset_index(drop=True)
        for i in range(1, len(recent_10)):
            if recent_10.iloc[i]['low'] > recent_10.iloc[i-1]['high']:
                has_gap = True
                break
                
        is_red_15 = (df['close'].tail(15) > df['open'].tail(15)).astype(int)
        has_streak = (is_red_15.rolling(window=4).sum() == 4).any()
        
        vol_ma5 = df['volume'].tail(6).iloc[:-1].mean()
        is_double_vol = (curr['volume'] > prev['volume'] * 1.8) or (curr['volume'] > vol_ma5 * 1.8)

        if has_limit_up_20 and has_gap and has_streak and is_double_vol:
            signal_tags.append("👑四星共振")
            priority = 100
            action = "STRONG BUY"
            
        # 多头排列检查
        elif prev['open'] < prev['close'] and curr['close'] > prev['close']: 
            if priority == 0: 
                action = "HOLD (持有)"
                priority = 10
                signal_tags.append("📈多头")

        if priority == 0: 
            return None

        return {
            "result": {
                "代码": code, 
                "名称": info['name'], 
                "所属行业": info['industry'],
                "现价": f"{curr['close']:.2f}", 
                "涨跌": f"{curr['pctChg']:.2f}%", 
                "获利筹码": f"{winner_rate:.1f}%",
                "风险评级": risk_level,
                "策略信号": " + ".join(signal_tags) if signal_tags else "无",
                "综合评级": action,
                "priority": priority
            },
            "alert": f"{info['name']}" if priority >= 90 else None,
            "option": f"{code} | {info['name']}"
        }

    def scan_market_optimized(self, code_list, max_price=None):
        results, alerts, valid_codes_list = [], [], []
        
        if not code_list:
            st.warning("股票列表为空！")
            return results, alerts, valid_codes_list
            
        progress_bar = st.progress(0, text=f"🚀 正在启动稳定扫描 (共 {len(code_list)} 只)...")
        total = len(code_list)
        
        for i, code in enumerate(code_list):
            if i % 2 == 0:
                progress_bar.progress((i + 1) / total, text=f"🔍 正在分析: {code} ({i+1}/{total}) | 已命中: {len(results)} 只")
            
            try:
                res = self._process_single_stock(code, max_price)
                if res:
                    results.append(res["result"])
                    if res["alert"]: 
                        alerts.append(res["alert"])
                    valid_codes_list.append(res["option"])
            except Exception as e:
                st.warning(f"处理 {code} 时出错: {e}")
                continue

        progress_bar.empty()
        return results, alerts, valid_codes_list

    def get_deep_data(self, code):
        try:
            bs.login()
            end = datetime.datetime.now().strftime("%Y-%m-%d")
            start = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
            
            rs = bs.query_history_k_data_plus(
                code, 
                "date,open,close,high,low,volume,peTTM,pbMRQ", 
                start_date=start, 
                end_date=end, 
                frequency="d", 
                adjustflag="3"
            )
            
            data = []
            while rs.next(): 
                data.append(rs.get_row_data())
                
            bs.logout()
                
            if not data: 
                return None
                
            df = pd.DataFrame(data, columns=["date", "open", "close", "high", "low", "volume", "peTTM", "pbMRQ"])
            cols = ['open', 'close', 'high', 'low', 'volume', 'peTTM', 'pbMRQ']
            
            for col in cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            df = df.dropna(subset=['close'])
            return df
            
        except Exception as e:
            try:
                bs.logout()
            except:
                pass
            return None

    def run_ai_prediction(self, df):
        if df is None or len(df) < 30: 
            return None
            
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
        if df is None or df.empty:
            return df
            
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
        if df is None or df.empty:
            return None
            
        df = df.copy()
        df['MA5'] = df['close'].rolling(5).mean()
        df['MA20'] = df['close'].rolling(20).mean()
        
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
            fig.add_trace(go.Scatter(
                x=buy_points['date'], 
                y=buy_points['low']*0.98, 
                mode='markers+text', 
                marker=dict(symbol='triangle-up', size=12, color='red'), 
                text='B', 
                textposition='bottom center', 
                name='买入'
            ))
        
        if not sell_points.empty:
            fig.add_trace(go.Scatter(
                x=sell_points['date'], 
                y=sell_points['high']*1.02, 
                mode='markers+text', 
                marker=dict(symbol='triangle-down', size=12, color='green'), 
                text='S', 
                textposition='top center', 
                name='卖出'
            ))

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

# 初始化session_state
if 'full_pool' not in st.session_state:
    st.session_state['full_pool'] = []
    
if 'scan_res' not in st.session_state:
    st.session_state['scan_res'] = []
    
if 'valid_options' not in st.session_state:
    st.session_state['valid_options'] = []
    
if 'alerts' not in st.session_state:
    st.session_state['alerts'] = []

if pool_mode == "手动输入":
    default_pool = "600519, 002131, 002312, 600580, 002594"
    target_pool_str = st.sidebar.text_area("监控股票池", default_pool, height=100)
    final_code_list = [code.strip() for code in target_pool_str.replace("，", ",").split(",") if code.strip()]
else:
    if st.sidebar.button(f"📥 加载 {pool_mode} 成分股"):
        with st.spinner("正在获取成分股..."):
            index_code = "zz500" if "中证500" in pool_mode else "hs300"
            stock_list = engine.get_index_stocks(index_code)
            if stock_list:
                st.session_state['full_pool'] = stock_list 
                st.sidebar.success(f"已加载全量 {len(stock_list)} 只股票")
            else:
                st.sidebar.error("获取成分股失败，请检查网络连接")
    
    if st.session_state['full_pool']:
        full_list = st.session_state['full_pool']
        final_code_list = full_list[:scan_limit] 
        st.sidebar.info(f"池内待扫: {len(final_code_list)} 只 (总库: {len(full_list)})")
    else:
        final_code_list = []

st.sidebar.markdown("---")
if st.sidebar.button("🚀 启动全策略扫描 (V45)", type="primary"):
    if not final_code_list:
        st.sidebar.error("请先加载股票池！")
    else:
        st.caption(f"当前筛选：价格 < {max_price_limit}元 | 剔除ST/科创/北交 | 模式：长连接稳定扫描")
        scan_res, alerts, valid_options = engine.scan_market_optimized(final_code_list, max_price=max_price_limit)
        st.session_state['scan_res'] = scan_res
        st.session_state['valid_options'] = valid_options
        st.session_state['alerts'] = alerts
        
        if not scan_res:
            st.info("扫描完成，但未找到符合条件的股票。请尝试放宽筛选条件。")

with st.expander("📖 **策略逻辑白皮书**", expanded=False):
    st.markdown("##### 🔍 核心策略定义")
    for k, v in STRATEGY_LOGIC.items(): 
        st.markdown(f"- **{k}**: {v}")

st.subheader(f"⚡ 扫描结果 (价格 < {max_price_limit}元)")

if st.session_state['scan_res']:
    results = st.session_state['scan_res']
    alerts = st.session_state['alerts']
    
    if alerts: 
        alert_names = "、".join(alerts[:5])  # 只显示前5个，避免太长
        st.success(f"🔥 发现 {len(alerts)} 只【主力高控盘】标的：**{alert_names}**")
    
    df_scan = pd.DataFrame(results)
    
    if df_scan.empty:
        st.warning("⚠️ 扫描完成，无符合条件的股票。")
    else:
        # 确保priority列存在
        if 'priority' in df_scan.columns:
            df_scan = df_scan.sort_values(by="priority", ascending=False)
        
        # 格式化数据
        display_df = df_scan.copy()
        if 'priority' in display_df.columns:
            display_df = display_df.drop(columns=['priority'])
            
        st.dataframe(
            display_df, 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "代码": st.column_config.TextColumn("代码"),
                "名称": st.column_config.TextColumn("名称"),
                "现价": st.column_config.NumberColumn("现价", format="%.2f"),
                "涨跌": st.column_config.TextColumn("涨跌"),
                "获利筹码": st.column_config.TextColumn("获利筹码"),
                "风险评级": st.column_config.TextColumn("风险评级", help="基于乖离率计算"),
                "策略信号": st.column_config.TextColumn("策略信号", help=STRATEGY_TIP, width="large"),
                "综合评级": st.column_config.TextColumn("综合评级", help=ACTION_TIP, width="medium"),
            }
        )
else:
    st.info("👈 请在左侧加载股票 -> 点击'启动全策略扫描'")

st.divider()

if st.session_state['valid_options']:
    st.subheader("🧠 深度分析")
    target = st.selectbox("选择目标进行深度分析", st.session_state['valid_options'])
    
    target_code = target.split("|")[0].strip()
    target_name = target.split("|")[1].strip()

    if st.button(f"🚀 立即分析 {target_name}"):
        with st.spinner("AI 正在推演未来变盘点..."):
            df = engine.get_deep_data(target_code)
            if df is not None and not df.empty:
                df = engine.calc_indicators(df)
                future_info = engine.run_ai_prediction(df)
                
                if future_info:
                    last = df.iloc[-1]
                    col1, col2, col3 = st.columns(3)
                    col1.metric("当前价格", f"¥{last['close']:.2f}")
                    
                    delta_value = future_info['pred_price'] - last['close']
                    delta_color = "normal" if delta_value > 0 else "inverse"
                    col2.metric("AI预测明日", f"¥{future_info['pred_price']:.2f}", 
                               delta=f"{delta_value:.2f}", 
                               delta_color=delta_color)
                               
                    pe = last.get('peTTM', 0)
                    col3.metric("PE估值", f"{pe:.1f}")
                    
                    # 根据颜色显示不同的消息框
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
                    st.warning("数据不足，无法进行AI预测")
                    
                # 绘制K线图
                fig = engine.plot_professional_kline(df, target_name)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    st.info("💡 **图例**: 🔺红色B=金叉买点 | 🔻绿色S=死叉卖点 (仅供辅助参考)")
                else:
                    st.warning("无法生成K线图，数据可能不足")
            else:
                st.error("无法获取股票数据，请稍后重试")