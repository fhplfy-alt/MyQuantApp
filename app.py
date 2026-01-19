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
        pass

    def clean_code(self, code):
        code = str(code).strip()
        if not (code.startswith('sh.') or code.startswith('sz.')):
            return f"sh.{code}" if code.startswith('6') else f"sz.{code}"
        return code

    def is_valid(self, code, name):
        if "sh.688" in code: 
            return False  # 科创板
        if "bj." in code or code.startswith("sz.8") or code.startswith("sz.4"): 
            return False  # 北交所和退市板块
        if "ST" in name or "*" in name: 
            return False  # ST股票
        return True

    def get_index_stocks(self, index_type="zz500"):
        try:
            bs.login()
            stocks = []
            if index_type == "hs300":
                rs = bs.query_hs300_stocks()
            else:
                rs = bs.query_zz500_stocks()
            
            while rs.next(): 
                stock_code = rs.get_row_data()[1]
                stocks.append(stock_code)
            return stocks
        except Exception as e:
            st.warning(f"获取指数成分股时出错: {e}")
            return []
        finally:
            try:
                bs.logout()
            except:
                pass

    def calc_winner_rate(self, df, current_price):
        """计算获利筹码比例 - 简化版"""
        if df.empty or current_price <= 0:
            return 0.0
            
        # 简化计算：假设股价在近期低点和高点之间均匀分布
        recent_low = df['low'].min()
        recent_high = df['high'].max()
        
        if recent_high == recent_low:
            return 50.0
            
        # 当前价格在历史区间中的位置
        position = (current_price - recent_low) / (recent_high - recent_low) * 100
        # 调整公式：当前价格越高，获利筹码比例越低
        winner_rate = max(0, min(100, 100 - position))
        
        return winner_rate

    def calc_risk_level(self, price, ma5, ma20):
        if ma5 == 0: 
            return "未知"
        bias = (price - ma5) / ma5 * 100
        if bias > 15: 
            return "High (高危)"
        elif bias < -10: 
            return "Med (弱势)"
        elif price < ma20: 
            return "Med (破位)"
        else: 
            return "Low (安全)"

    def _process_single_stock(self, code, max_price=None):
        """处理单个股票 - 修复版"""
        code = self.clean_code(code)
        
        # 设置时间范围
        end_date = datetime.datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.datetime.now() - datetime.timedelta(days=180)).strftime("%Y-%m-%d")
        
        data = []
        info = {'name': code, 'industry': '-', 'ipoDate': '2000-01-01'}
        
        try:
            # 获取股票基本信息
            bs.login()
            
            # 获取基础信息
            rs_info = bs.query_stock_basic(code=code)
            if rs_info.error_code != '0': 
                return None
                
            if rs_info.next():
                row = rs_info.get_row_data()
                info['name'] = row[1] if row[1] else code
                info['ipoDate'] = row[2] if row[2] else '2000-01-01'
            
            # 检查是否有效
            if not self.is_valid(code, info['name']): 
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
            
            while rs.next(): 
                data.append(rs.get_row_data())
                
        except Exception as e:
            return None
        finally:
            try:
                bs.logout()
            except:
                pass

        if not data or len(data) < 60:
            return None
            
        try:
            # 创建DataFrame
            df = pd.DataFrame(data, columns=["date", "open", "close", "high", "low", "volume", "pctChg", "turn"])
            
            # 转换数据类型
            numeric_cols = ["open", "close", "high", "low", "volume", "pctChg", "turn"]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 清理数据
            df = df.dropna(subset=numeric_cols)
            df = df.reset_index(drop=True)
            
        except Exception as e:
            return None
            
        if len(df) < 60:
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
        
        risk_level = self.calc_risk_level(curr['close'], df['MA5'].iloc[-1], df['MA20'].iloc[-1])
        
        # 策略信号检测
        signal_tags = []
        priority = 0
        action = "WAIT (观望)"
        
        # 策略1: 多头排列 (放宽条件)
        if curr['close'] > df['MA5'].iloc[-1] and df['MA5'].iloc[-1] > df['MA20'].iloc[-1]:
            signal_tags.append("📈多头排列")
            priority = 20
            action = "HOLD (持有)"
        
        # 策略2: 温和吸筹 (放宽条件)
        if len(df) >= 3:
            recent_3 = df.tail(3)
            is_3_up = all(recent_3['pctChg'] > 0)
            sum_3_rise = recent_3['pctChg'].sum()
            
            if is_3_up and sum_3_rise <= 8 and winner_rate > 50:  # 放宽条件
                signal_tags.append("🔴温和吸筹")
                priority = max(priority, 40)
                action = "BUY (低吸)"
        
        # 策略3: 换手锁仓 (放宽条件)
        if len(df) >= 2:
            recent_2 = df.tail(2)
            is_high_turn = all(recent_2['turn'] > 3)  # 降低换手率要求
            
            if is_high_turn and winner_rate > 60:  # 放宽获利筹码要求
                signal_tags.append("🔥换手锁仓")
                priority = max(priority, 50)
                action = "BUY (博弈)"
        
        # 策略4: 金叉信号
        if len(df) >= 2:
            curr_ma5 = df['MA5'].iloc[-1]
            curr_ma20 = df['MA20'].iloc[-1]
            prev_ma5 = df['MA5'].iloc[-2]
            prev_ma20 = df['MA20'].iloc[-2]
            
            if prev_ma5 <= prev_ma20 and curr_ma5 > curr_ma20:
                signal_tags.append("🚀金叉突破")
                priority = max(priority, 60)
                action = "BUY (博弈)"
        
        # 策略5: 妖股基因 (放宽条件)
        df_60 = df.tail(60)
        limit_up_count = len(df_60[df_60['pctChg'] > 9.0])  # 降低涨停要求
        
        try:
            ipo_date = datetime.datetime.strptime(info['ipoDate'], "%Y-%m-%d")
            days_listed = (datetime.datetime.now() - ipo_date).days
        except:
            days_listed = 365
            
        if limit_up_count >= 2 and winner_rate > 70 and days_listed > 30:  # 放宽条件
            signal_tags.append("🐲潜力龙头")
            priority = max(priority, 70)
            action = "STRONG BUY"
        
        # 策略6: 量价齐升
        if curr['volume'] > df['volume'].mean() * 1.5 and curr['pctChg'] > 2:
            signal_tags.append("📊量价齐升")
            priority = max(priority, 30)
            action = "BUY (低吸)"
        
        # 如果没有任何信号，返回None
        if priority == 0:
            return None
            
        # 返回结果
        return {
            "result": {
                "代码": code,
                "名称": info['name'],
                "所属行业": info['industry'],
                "现价": f"{curr['close']:.2f}",
                "涨跌": f"{curr['pctChg']:.2f}%",
                "获利筹码": f"{winner_rate:.1f}%",
                "风险评级": risk_level,
                "策略信号": " + ".join(signal_tags) if signal_tags else "趋势跟踪",
                "综合评级": action,
                "priority": priority
            },
            "alert": f"{info['name']}" if priority >= 70 else None,
            "option": f"{code} | {info['name']}"
        }

    def scan_market_optimized(self, code_list, max_price=None):
        """扫描市场 - 优化版"""
        results = []
        alerts = []
        valid_codes_list = []
        
        if not code_list:
            st.warning("股票列表为空！")
            return results, alerts, valid_codes_list
            
        total = len(code_list)
        if total == 0:
            return results, alerts, valid_codes_list
            
        # 创建进度条
        progress_bar = st.progress(0, text=f"🚀 正在扫描 {total} 只股票...")
        
        for i, code in enumerate(code_list):
            # 更新进度
            progress_value = (i + 1) / total
            progress_bar.progress(progress_value, 
                                 text=f"🔍 扫描中: {code} ({i+1}/{total}) | 已命中: {len(results)} 只")
            
            try:
                # 处理单个股票
                res = self._process_single_stock(code, max_price)
                
                if res:
                    results.append(res["result"])
                    if res["alert"]: 
                        alerts.append(res["alert"])
                    valid_codes_list.append(res["option"])
                    
            except Exception as e:
                # 跳过错误，继续扫描
                continue
                
            # 添加小延迟避免请求过快
            time.sleep(0.05)
        
        # 清理进度条
        progress_bar.empty()
        
        return results, alerts, valid_codes_list

    def get_deep_data(self, code):
        """获取深度数据"""
        try:
            bs.login()
            
            end_date = datetime.datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
            
            rs = bs.query_history_k_data_plus(
                code, 
                "date,open,close,high,low,volume,turn,pctChg", 
                start_date=start_date, 
                end_date=end_date, 
                frequency="d", 
                adjustflag="3"
            )
            
            data = []
            while rs.next(): 
                data.append(rs.get_row_data())
                
            if not data:
                return None
                
            df = pd.DataFrame(data, columns=["date", "open", "close", "high", "low", "volume", "turn", "pctChg"])
            
            # 转换数据类型
            numeric_cols = ["open", "close", "high", "low", "volume", "turn", "pctChg"]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna()
            return df
            
        except Exception as e:
            return None
        finally:
            try:
                bs.logout()
            except:
                pass

    def run_ai_prediction(self, df):
        """AI预测"""
        if df is None or len(df) < 30:
            return None
            
        try:
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
            
            if slope > 0.03:
                hint_title = "🚀 上升趋势"
                hint_desc = f"趋势向上，预计 {future_dates[1]} 到达 ¥{pred_prices[1]:.2f}"
                action = "建议：持有或逢低买入"
                color = "red"
            elif slope > 0:
                hint_title = "📈 缓慢上行"
                hint_desc = f"温和上涨，预计 {future_dates[1]} 到达 ¥{pred_prices[1]:.2f}"
                action = "建议：耐心持股"
                color = "orange"
            elif slope < -0.03:
                hint_title = "📉 下跌趋势"
                hint_desc = f"趋势向下，建议观望"
                action = "建议：控制风险"
                color = "green"
            else:
                hint_title = "⚖️ 横盘震荡"
                hint_desc = f"震荡整理，等待方向选择"
                action = "建议：观望等待"
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
        except:
            return None

    def calc_indicators(self, df):
        """计算技术指标"""
        if df is None or df.empty:
            return df
            
        df = df.copy()
        df['MA5'] = df['close'].rolling(5).mean()
        df['MA20'] = df['close'].rolling(20).mean()
        df['MA60'] = df['close'].rolling(60).mean()
        
        return df

    def plot_professional_kline(self, df, title):
        """绘制K线图"""
        if df is None or df.empty:
            return None
            
        df = self.calc_indicators(df)
        
        fig = go.Figure()
        
        # K线
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
        
        # 均线
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['MA5'],
            name='MA5',
            line=dict(color='orange', width=1)
        ))
        
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['MA20'],
            name='MA20',
            line=dict(color='blue', width=1)
        ))
        
        fig.update_layout(
            title=f"{title} - K线图",
            xaxis_title="日期",
            yaxis_title="价格",
            xaxis_rangeslider_visible=False,
            height=500
        )
        
        return fig

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

st.sidebar.header("🕹️ 控制台")

# 价格上限
max_price_limit = st.sidebar.slider("💰 价格上限 (元)", 3.0, 100.0, 50.0, 1.0)

# 选股范围
pool_mode = st.sidebar.radio("🔎 选股范围:", 
                            ["中证500 (中小盘)", "沪深300 (大盘)", "手动输入", "测试模式"])

# 扫描数量
scan_limit = st.sidebar.slider("🔢 扫描数量", 10, 500, 100, 10)

st.sidebar.markdown("---")

# 股票池管理
if pool_mode == "手动输入":
    default_pool = """600519,000858,000333,002415,300750,600036,601318,000001,600030,000002"""
    target_pool_str = st.sidebar.text_area("📝 输入股票代码 (用逗号分隔)", default_pool, height=120)
    final_code_list = [code.strip() for code in target_pool_str.replace("，", ",").split(",") if code.strip()]
    
    if final_code_list:
        st.sidebar.success(f"✅ 已加载 {len(final_code_list)} 只股票")
    
elif pool_mode == "测试模式":
    # 测试用的股票池
    test_codes = [
        "600519", "000858", "000333", "002415", "300750",
        "600036", "601318", "000001", "600030", "000002",
        "300059", "000063", "002594", "600887", "600276"
    ]
    final_code_list = test_codes[:scan_limit]
    st.sidebar.info(f"🧪 测试模式: {len(final_code_list)} 只测试股票")
    
else:
    # 指数成分股模式
    if st.sidebar.button(f"📥 加载 {pool_mode} 成分股", type="primary"):
        with st.spinner(f"正在获取{pool_mode}成分股..."):
            index_code = "zz500" if "中证500" in pool_mode else "hs300"
            stock_list = engine.get_index_stocks(index_code)
            
            if stock_list:
                st.session_state['full_pool'] = stock_list
                st.sidebar.success(f"✅ 已加载 {len(stock_list)} 只成分股")
            else:
                st.sidebar.error("❌ 获取成分股失败，请检查网络")
    
    if st.session_state['full_pool']:
        full_list = st.session_state['full_pool']
        final_code_list = full_list[:scan_limit]
        st.sidebar.info(f"📊 池内待扫: {len(final_code_list)} 只 (总库: {len(full_list)})")
    else:
        final_code_list = []

# 扫描按钮
st.sidebar.markdown("---")
if st.sidebar.button("🚀 启动智能扫描", type="primary", use_container_width=True):
    if not final_code_list:
        st.sidebar.error("❌ 请先加载股票池！")
    else:
        with st.spinner("正在准备扫描..."):
            st.caption(f"📊 当前筛选：价格 < {max_price_limit}元 | 扫描数量: {len(final_code_list)}")
            
            # 执行扫描
            scan_res, alerts, valid_options = engine.scan_market_optimized(
                final_code_list, 
                max_price=max_price_limit
            )
            
            # 保存结果到session_state
            st.session_state['scan_res'] = scan_res
            st.session_state['valid_options'] = valid_options
            st.session_state['alerts'] = alerts
            
            # 显示扫描统计
            if scan_res:
                st.success(f"✅ 扫描完成！发现 {len(scan_res)} 只符合条件的股票")
                if alerts:
                    st.info(f"🔥 发现 {len(alerts)} 只高潜力标的")
            else:
                st.warning("⚠️ 扫描完成，但未发现符合条件的股票")

# 策略说明
with st.expander("📖 策略说明", expanded=True):
    st.markdown("### 🎯 当前策略说明")
    st.markdown("""
    本系统采用**多策略组合**扫描，主要包括：
    
    1. **📈 多头排列** - 趋势跟踪策略
    2. **🔴 温和吸筹** - 主力吸筹识别
    3. **🔥 换手锁仓** - 高换手博弈机会
    4. **🚀 金叉突破** - 技术指标信号
    5. **🐲 潜力龙头** - 强势股识别
    6. **📊 量价齐升** - 量价配合机会
    
    **⚠️ 注意**：扫描结果仅供参考，投资需谨慎！
    """)

# 显示扫描结果
st.subheader(f"⚡ 扫描结果 (价格 < {max_price_limit}元)")

if st.session_state['scan_res']:
    results = st.session_state['scan_res']
    alerts = st.session_state['alerts']
    
    # 显示高潜力标的
    if alerts:
        alert_display = "、".join(alerts[:3])  # 只显示前3个
        if len(alerts) > 3:
            alert_display += f" 等{len(alerts)}只"
        st.success(f"🎯 **高潜力标的**: {alert_display}")
    
    # 转换为DataFrame
    df_scan = pd.DataFrame(results)
    
    if not df_scan.empty:
        # 按优先级排序
        if 'priority' in df_scan.columns:
            df_scan = df_scan.sort_values(by="priority", ascending=False)
        
        # 显示数据
        st.dataframe(
            df_scan,
            use_container_width=True,
            hide_index=True,
            column_config={
                "代码": st.column_config.TextColumn("代码", width="small"),
                "名称": st.column_config.TextColumn("名称", width="medium"),
                "现价": st.column_config.NumberColumn("现价", format="%.2f", width="small"),
                "涨跌": st.column_config.TextColumn("涨跌", width="small"),
                "获利筹码": st.column_config.TextColumn("获利筹码", width="small"),
                "风险评级": st.column_config.TextColumn("风险评级", width="small"),
                "策略信号": st.column_config.TextColumn("策略信号", width="large", help=STRATEGY_TIP),
                "综合评级": st.column_config.TextColumn("操作建议", width="medium", help=ACTION_TIP),
                "priority": None  # 隐藏优先级列
            }
        )
        
        # 显示统计信息
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📈 扫描总数", len(final_code_list))
        with col2:
            st.metric("✅ 命中数量", len(results))
        with col3:
            hit_rate = (len(results) / len(final_code_list) * 100) if final_code_list else 0
            st.metric("🎯 命中率", f"{hit_rate:.1f}%")
    else:
        st.warning("⚠️ 暂无符合条件的股票")
else:
    st.info("👈 请在左侧配置参数并启动扫描")

# 深度分析部分
st.divider()
st.subheader("🧠 深度分析")

if st.session_state['valid_options']:
    target = st.selectbox("选择股票进行深度分析", 
                         st.session_state['valid_options'],
                         help="选择扫描结果中的股票进行详细分析")
    
    if target:
        target_code = target.split("|")[0].strip()
        target_name = target.split("|")[1].strip()
        
        if st.button(f"🔍 分析 {target_name}", type="primary"):
            with st.spinner("正在分析中..."):
                # 获取数据
                df = engine.get_deep_data(target_code)
                
                if df is not None and not df.empty:
                    # 显示基本信息
                    col1, col2, col3 = st.columns(3)
                    latest = df.iloc[-1]
                    
                    with col1:
                        st.metric("当前价格", f"¥{latest['close']:.2f}")
                    with col2:
                        st.metric("今日涨跌", f"{latest['pctChg']:.2f}%")
                    with col3:
                        avg_vol = df['volume'].mean()
                        vol_ratio = latest['volume'] / avg_vol if avg_vol > 0 else 1
                        st.metric("成交量比", f"{vol_ratio:.1f}倍")
                    
                    # AI预测
                    future_info = engine.run_ai_prediction(df)
                    if future_info:
                        st.markdown(f"### 🤖 AI预测: {future_info['title']}")
                        st.markdown(future_info['desc'])
                        st.markdown(f"**{future_info['action']}**")
                        
                        # 显示未来3日预测
                        st.markdown("#### 📅 未来3日预测")
                        pred_cols = st.columns(3)
                        for i in range(3):
                            with pred_cols[i]:
                                st.metric(future_info['dates'][i], 
                                         f"¥{future_info['prices'][i]:.2f}")
                    
                    # K线图
                    st.markdown("### 📊 K线分析")
                    fig = engine.plot_professional_kline(df, target_name)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption("💡 提示: 橙色线为5日均线，蓝色线为20日均线")
                    else:
                        st.warning("无法生成K线图")
                        
                else:
                    st.error("无法获取该股票的数据")
else:
    st.info("👆 请先完成扫描以选择分析目标")

# 底部说明
st.divider()
st.caption("""
💡 **使用提示**: 
1. 首次使用时建议选择"测试模式"或"手动输入"模式
2. 可以调整价格上限来筛选不同价位的股票
3. 扫描结果每天会有所变化，建议定期更新
4. 投资有风险，决策需谨慎
""")