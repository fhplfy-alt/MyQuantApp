import datetime
import time
import streamlit as st
import baostock as bs
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

# 策略说明配置
STRATEGY_DESC = {
    "🐲 妖股基因": "近60日涨停≥3次 + 获利筹码>80% + 上市>30天",
    "🔥 换手锁仓": "连续2日换手率>5% + 获利筹码>70%",
    "🔴 温和吸筹": "3连阳且累计涨幅<5% + 获利筹码>62%",
    "📈 多头排列": "昨日收阳 且 今日收盘价 > 昨日收盘价",
    "👑 四星共振": "20日有涨停 + 10日有跳空 + 15日有4连阳 + 放量1.8倍"
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
        # ========== 新增：买卖操作详细提示 ==========
        buy_reason = ""
        sell_warning = ""
        position_suggestion = "0% (空仓)"  # 仓位建议
        stop_loss_price = round(curr['close'] * 0.95, 2)  # 止损价（默认5%）
        take_profit_price = 0  # 止盈价
        
        is_3_up = all(df['pctChg'].tail(3) > 0)
        sum_3_rise = df['pctChg'].tail(3).sum()
        if (is_3_up and sum_3_rise <= 5 and winner_rate > 62):
            signal_tags.append("🔴温和吸筹")
            priority = max(priority, 60)
            action = "BUY (低吸)"
            buy_reason = "3连阳且累计涨幅温和，获利筹码充足，低位吸筹安全边际高"
            position_suggestion = "20-30% (轻仓)"
            take_profit_price = round(curr['close'] * 1.10, 2)  # 止盈10%

        is_high_turn = all(df['turn'].tail(2) > 5) 
        if is_high_turn and winner_rate > 70:
            signal_tags.append("🔥换手锁仓")
            priority = max(priority, 70)
            action = "BUY (博弈)"
            buy_reason = "连续高换手洗盘，获利筹码占比高，资金锁仓意愿强"
            position_suggestion = "30-50% (中仓)"
            take_profit_price = round(curr['close'] * 1.15, 2)  # 止盈15%
            stop_loss_price = round(curr['close'] * 0.93, 2)  # 止损7%

        df_60 = df.tail(60)
        limit_up_60 = len(df_60[df_60['pctChg'] > 9.5])
        if limit_up_60 >= 3 and winner_rate > 80 and days_listed > 30:
            signal_tags.append("🐲妖股基因")
            priority = max(priority, 90)
            action = "STRONG BUY (重仓)"
            buy_reason = "近60日涨停次数多，获利筹码高度集中，妖股特征明显"
            position_suggestion = "50-70% (重仓)"
            take_profit_price = round(curr['close'] * 1.20, 2)  # 止盈20%
            stop_loss_price = round(curr['close'] * 0.90, 2)  # 止损10%

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
            action = "STRONG BUY (满仓)"
            buy_reason = "四星共振形态形成，量价齐升，短期爆发概率极高"
            position_suggestion = "70-100% (满仓)"
            take_profit_price = round(curr['close'] * 1.25, 2)  # 止盈25%
            stop_loss_price = round(curr['close'] * 0.88, 2)  # 止损12%
        elif prev['open'] < prev['close'] and curr['close'] > prev['close']: 
             if priority == 0: 
                 action = "HOLD (持有)"
                 priority = 10
                 signal_tags.append("📈多头排列")
                 buy_reason = "多头趋势形成，可继续持有"
                 position_suggestion = "持有当前仓位"
        
        # ========== 新增：卖出信号判断 ==========
        # 高危风险触发卖出
        if risk_level == "High (高危)":
            action = "SELL (卖出)"
            sell_warning = "股价偏离5日均线过远，短期回调风险极大，建议立即卖出"
            position_suggestion = "0% (清仓)"
        # 破位触发卖出
        elif risk_level == "Med (破位)":
            action = "SELL (减仓)"
            sell_warning = "股价跌破20日均线，趋势走弱，建议减仓或清仓"
            position_suggestion = "0-20% (轻仓观望)"
        
        if priority == 0: return None

        return {
            "result": {
                "代码": code, 
                "名称": info['name'], 
                "所属行业": info['industry'],
                "现价": curr['close'], 
                "涨跌": f"{curr['pctChg']:.2f}%", 
                "获利筹码": round(winner_rate, 2),
                "风险评级": risk_level,
                "策略信号": " + ".join(signal_tags),
                "综合操作": action,
                "操作理由": buy_reason if buy_reason else sell_warning if sell_warning else "暂无明确操作信号",
                "仓位建议": position_suggestion,
                "止损价": stop_loss_price,
                "止盈价": take_profit_price if take_profit_price > 0 else "暂无",
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

# ========== 新增：页面标题和操作提示 ==========
st.title("📊 智能股票买卖决策系统")
st.markdown("### 📌 核心功能：基于多维度策略自动生成买卖信号、仓位建议、止盈止损价")
st.markdown("---")

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

# ========== 新增：扫描结果展示优化（突出买卖提示） ==========
if st.session_state['scan_res']:
    st.subheader("📋 股票买卖决策结果")
    
    # 按优先级排序（高优先级在前）
    sorted_res = sorted(st.session_state['scan_res'], key=lambda x: x['priority'], reverse=True)
    
    # 分类展示：买入/持有/卖出
    buy_stocks = [s for s in sorted_res if "BUY" in s['综合操作']]
    hold_stocks = [s for s in sorted_res if "HOLD" in s['综合操作']]
    sell_stocks = [s for s in sorted_res if "SELL" in s['综合操作']]
    
    # 买入信号展示
    if buy_stocks:
        st.markdown("### 🟢 买入信号（按优先级排序）")
        for stock in buy_stocks:
            # 不同优先级用不同颜色卡片
            if stock['priority'] >= 90:
                card_color = "#d4edda"  # 深绿（重仓/满仓）
            elif stock['priority'] >= 70:
                card_color = "#e8f5e9"  # 中绿（中仓）
            else:
                card_color = "#f1f8e9"  # 浅绿（轻仓）
            
            with st.container():
                st.markdown(f"""
                <div style="background-color:{card_color};padding:15px;border-radius:8px;margin-bottom:10px;">
                    <h4 style="margin:0;color:#2e7d32;">{stock['名称']} ({stock['代码']}) - {stock['综合操作']}</h4>
                    <p style="margin:5px 0;"><strong>所属行业：</strong>{stock['所属行业']}</p>
                    <p style="margin:5px 0;"><strong>现价：</strong>¥{stock['现价']:.2f} | <strong>涨跌：</strong>{stock['涨跌']} | <strong>获利筹码：</strong>{stock['获利筹码']}%</p>
                    <p style="margin:5px 0;"><strong>风险评级：</strong>{stock['风险评级']} | <strong>策略信号：</strong>{stock['策略信号']}</p>
                    <p style="margin:5px 0;"><strong>操作理由：</strong>{stock['操作理由']}</p>
                    <p style="margin:5px 0;"><strong>仓位建议：</strong>{stock['仓位建议']} | <strong>止损价：</strong>¥{stock['止损价']} | <strong>止盈价：</strong>¥{stock['止盈价']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # 持有信号展示
    if hold_stocks:
        st.markdown("### 🟡 持有信号")
        for stock in hold_stocks:
            card_color = "#fff3cd"
            with st.container():
                st.markdown(f"""
                <div style="background-color:{card_color};padding:15px;border-radius:8px;margin-bottom:10px;">
                    <h4 style="margin:0;color:#856404;">{stock['名称']} ({stock['代码']}) - {stock['综合操作']}</h4>
                    <p style="margin:5px 0;"><strong>所属行业：</strong>{stock['所属行业']}</p>
                    <p style="margin:5px 0;"><strong>现价：</strong>¥{stock['现价']:.2f} | <strong>涨跌：</strong>{stock['涨跌']} | <strong>获利筹码：</strong>{stock['获利筹码']}%</p>
                    <p style="margin:5px 0;"><strong>风险评级：</strong>{stock['风险评级']} | <strong>策略信号：</strong>{stock['策略信号']}</p>
                    <p style="margin:5px 0;"><strong>操作理由：</strong>{stock['操作理由']}</p>
                    <p style="margin:5px 0;"><strong>仓位建议：</strong>{stock['仓位建议']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # 卖出信号展示
    if sell_stocks:
        st.markdown("### 🔴 卖出信号")
        for stock in sell_stocks:
            card_color = "#f8d7da"
            with st.container():
                st.markdown(f"""
                <div style="background-color:{card_color};padding:15px;border-radius:8px;margin-bottom:10px;">
                    <h4 style="margin:0;color:#721c24;">{stock['名称']} ({stock['代码']}) - {stock['综合操作']}</h4>
                    <p style="margin:5px 0;"><strong>所属行业：</strong>{stock['所属行业']}</p>
                    <p style="margin:5px 0;"><strong>现价：</strong>¥{stock['现价']:.2f} | <strong>涨跌：</strong>{stock['涨跌']} | <strong>获利筹码：</strong>{stock['获利筹码']}%</p>
                    <p style="margin:5px 0;"><strong>风险评级：</strong>{stock['风险评级']} | <strong>策略信号：</strong>{stock['策略信号']}</p>
                    <p style="margin:5px 0;"><strong>操作理由：</strong>{stock['操作理由']}</p>
                    <p style="margin:5px 0;"><strong>仓位建议：</strong>{stock['仓位建议']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # 导出功能（可选）
    if st.button("📤 导出买卖决策结果为Excel"):
        df_res = pd.DataFrame(sorted_res)
        # 只保留关键列
        df_export = df_res[['代码', '名称', '所属行业', '现价', '涨跌', '获利筹码', '风险评级', '策略信号', '综合操作', '操作理由', '仓位建议', '止损价', '止盈价']]
        st.download_button(
            label="点击下载",
            data=df_export.to_csv(index=False, encoding='utf-8-sig'),
            file_name=f"股票买卖决策_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
else:
    st.info("💡 请点击左侧【启动全策略扫描】按钮获取买卖决策结果")

# ========== 新增：个股深度分析（带买卖提示） ==========
if st.session_state['valid_options']:
    st.markdown("---")
    st.subheader("🔍 个股深度分析")
    selected_stock = st.selectbox("选择股票", st.session_state['valid_options'])
    if selected_stock:
        code = selected_stock.split(" | ")[0]
        df = engine.get_deep_data(code)
        if df is not None:
            # 绘制K线图
            fig = engine.plot_professional_kline(df, selected_stock.split(" | ")[1])
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # AI预测
            ai_pred = engine.run_ai_prediction(df)
            if ai_pred:
                st.markdown(f"""
                <div style="background-color:#f0f8ff;padding:10px;border-radius:5px;margin-top:10px;">
                    <h5 style="margin:0;color:#0277bd;">{ai_pred['title']}</h5>
                    <p style="margin:5px 0;">{ai_pred['desc']}</p>
                    <p style="margin:5px 0;"><strong>操作建议：</strong>{ai_pred['action']}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("无法获取该股票的深度数据")
