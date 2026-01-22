import streamlit as st
import baostock as bs
import datetime
import pandas as pd
import numpy as np
import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import plotly.graph_objects as go

# ==========================================
# 1. 策略配置
# ==========================================
STRATEGY_DESC = {
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
        self.MAX_WORKERS = 5  # 并发数控制，避免接口限流
        self.PROCESS_TIMEOUT = 10  # 单只股票处理超时时间(秒)
    
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
        except Exception as e:
            st.warning(f"获取全市场股票失败: {str(e)}")
            try:
                bs.logout()
            except:
                pass
            return []

    def get_index_stocks(self, index_type="zz500"):
        """获取指数成分股"""
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
            st.warning(f"获取{index_type}成分股失败: {str(e)}")
        finally: 
            bs.logout()
        return stocks[:self.MAX_SCAN_LIMIT]

    def calc_winner_rate(self, df, current_price):
        if df.empty: return 0.0
        total_vol = df['volume'].sum()
        if total_vol == 0: return 0.0
        profit_vol = df[df['close'] < current_price]['volume'].sum()
        return round((profit_vol / total_vol) * 100, 2)

    def calc_risk_level(self, price, ma5, ma20):
        if ma5 == 0 or pd.isna(ma5) or pd.isna(ma20): 
            return "未知"
        bias = (price - ma5) / ma5 * 100
        if bias > 15: 
            return "High (高危)"
        elif price < ma20: 
            return "Med (破位)"
        else: 
            return "Low (安全)"

    def _process_single_stock(self, code, max_price=None):
        """处理单只股票（核心逻辑，增加全量异常捕获）"""
        code = self.clean_code(code)
        end = datetime.datetime.now().strftime("%Y-%m-%d")
        start = (datetime.datetime.now() - datetime.timedelta(days=150)).strftime("%Y-%m-%d")
        
        data = []
        info = {'name': code, 'industry': '-', 'ipoDate': '2000-01-01'}
        
        try:
            # 获取基础信息
            rs_info = bs.query_stock_basic(code=code)
            if rs_info.error_code != '0': 
                return None 
            if rs_info.next():
                row = rs_info.get_row_data()
                info['name'] = row[1] if len(row)>=2 else code
                info['ipoDate'] = row[2] if len(row)>=3 else "2000-01-01"
            
            # 获取行业信息
            rs_ind = bs.query_stock_industry(code)
            if rs_ind.next():
                ind_row = rs_ind.get_row_data()
                info['industry'] = ind_row[3] if len(ind_row)>=4 else "-"
            
            # 过滤无效股票
            if not self.is_valid(code, info['name']): 
                return None
            
            # 获取K线数据
            rs = bs.query_history_k_data_plus(
                code, 
                "date,open,close,high,low,volume,pctChg,turn", 
                start_date=start, 
                frequency="d", 
                adjustflag="3"
            )
            if rs.error_code != '0':
                return None
            while rs.next(): 
                data.append(rs.get_row_data())
                
        except Exception as e:
            st.warning(f"{code} 基础数据获取失败: {str(e)}")
            return None

        if not data: 
            return None
        
        # 数据转换与清洗
        try:
            df = pd.DataFrame(
                data, 
                columns=["date", "open", "close", "high", "low", "volume", "pctChg", "turn"]
            )
            # 安全转换数值类型
            for col in ["open", "close", "high", "low", "volume", "pctChg", "turn"]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            # 清理脏数据
            df = df.dropna(subset=['close', 'volume'])
        except Exception as e:
            st.warning(f"{code} 数据转换失败: {str(e)}")
            return None
        
        if len(df) < 60: 
            return None

        # 价格过滤
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        if max_price is not None and curr['close'] > max_price: 
            return None

        # 核心指标计算
        winner_rate = self.calc_winner_rate(df, curr['close'])
        try: 
            ipo_date = datetime.datetime.strptime(info['ipoDate'], "%Y-%m-%d")
        except: 
            ipo_date = datetime.datetime(2000, 1, 1)
        days_listed = (datetime.datetime.now() - ipo_date).days

        # 均线计算
        df['MA5'] = df['close'].rolling(5).mean()
        df['MA20'] = df['close'].rolling(20).mean()
        risk_level = self.calc_risk_level(curr['close'], df['MA5'].iloc[-1], df['MA20'].iloc[-1])

        # 策略信号计算
        signal_tags = []
        priority = 0
        action = "WAIT (观望)"

        # 温和吸筹
        is_3_up = all(df['pctChg'].tail(3) > 0)
        sum_3_rise = df['pctChg'].tail(3).sum()
        if (is_3_up and sum_3_rise <= 5 and winner_rate > 62):
            signal_tags.append("🔴温和吸筹")
            priority = max(priority, 60)
            action = "BUY (低吸)"

        # 换手锁仓
        is_high_turn = all(df['turn'].tail(2) > 5) 
        if is_high_turn and winner_rate > 70:
            signal_tags.append("🔥换手锁仓")
            priority = max(priority, 70)
            action = "BUY (博弈)"

        # 妖股基因
        df_60 = df.tail(60)
        limit_up_60 = len(df_60[df_60['pctChg'] > 9.5])
        if limit_up_60 >= 3 and winner_rate > 80 and days_listed > 30:
            signal_tags.append("🐲妖股基因")
            priority = max(priority, 90)
            action = "STRONG BUY"

        # 四星共振
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
        # 多头排列
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
                "现价": round(curr['close'], 2), 
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
    
    def _process_single_stock_with_timeout(self, code, max_price=None):
        """带超时控制的单股票处理"""
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._process_single_stock, code, max_price)
                return future.result(timeout=self.PROCESS_TIMEOUT)
        except TimeoutError:
            st.warning(f"股票{code}处理超时，跳过")
            return None
        except Exception as e:
            st.warning(f"股票{code}处理异常: {str(e)}")
            return None

    def scan_market_optimized(self, code_list, max_price=None):
        """优化版扫描逻辑 - 并发处理+超时控制"""
        results, alerts, valid_codes_list = [], [], []
        
        # 登录验证
        lg = bs.login()
        if lg.error_code != '0':
            st.error("连接服务器失败，请检查网络！")
            return [], [], []

        # 数量限制
        if len(code_list) > self.MAX_SCAN_LIMIT:
            code_list = code_list[:self.MAX_SCAN_LIMIT]
            st.info(f"⚠️ 股票数量超过限制，已截取前{self.MAX_SCAN_LIMIT}只")

        total = len(code_list)
        if total == 0:
            bs.logout()
            return [], [], []
        
        # 进度条初始化
        progress_container = st.empty()
        progress_bar = progress_container.progress(0, text=f"🚀 正在启动稳定扫描 (共 {total} 只)...")
        
        # 并发处理股票
        try:
            with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
                # 提交所有任务
                futures = {
                    executor.submit(self._process_single_stock_with_timeout, code, max_price): code 
                    for code in code_list
                }
                
                # 处理完成的任务
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    code = futures[future]
                    # 更新进度条
                    progress = (i + 1) / total
                    current_count = min(i + 1, total)
                    progress_bar.progress(
                        progress, 
                        text=f"🔍 正在分析: {code} ({current_count}/{total}) | 已命中: {len(results)} 只"
                    )
                    
                    # 获取结果
                    try:
                        res = future.result()
                        if res:
                            results.append(res["result"])
                            if res["alert"]: 
                                alerts.append(res["alert"])
                            valid_codes_list.append(res["option"])
                    except Exception as e:
                        continue
        except Exception as e:
            st.error(f"扫描过程异常: {str(e)}")
        finally:
            bs.logout()
            progress_container.empty()
        
        return results, alerts, valid_codes_list

    def get_deep_data(self, code):
        """获取深度数据 - 缩短时间范围+异常加固"""
        try:
            bs.login()
            end = datetime.datetime.now().strftime("%Y-%m-%d")
            start = (datetime.datetime.now() - datetime.timedelta(days=180)).strftime("%Y-%m-%d")
            
            rs = bs.query_history_k_data_plus(
                code, 
                "date,open,close,high,low,volume",
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
            for col in ["open", "close", "high", "low", "volume"]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna(subset=['close', 'volume'])
            
            if len(df) < 20:
                return None
                
            return df
            
        except Exception as e:
            st.warning(f"获取{code}深度数据失败: {str(e)}")
            try:
                bs.logout()
            except:
                pass
            return None

    def run_ai_prediction(self, df):
        """AI预测 - 异常加固"""
        if df is None or len(df) < 20:
            return {
                "dates": ["明日", "后日", "大后日"],
                "prices": [0, 0, 0],
                "pred_price": 0,
                "title": "⚠️ 数据不足",
                "desc": "当前数据不足以进行准确预测",
                "action": "建议：补充数据后重试",
                "color": "blue"
            }
            
        try:
            recent = df.tail(20).reset_index(drop=True)
            X = np.array(recent.index).reshape(-1, 1)
            y = recent['close'].values
            
            if len(y) < 5 or np.isnan(y).any():
                return {
                    "dates": ["明日", "后日", "大后日"],
                    "prices": [0, 0, 0],
                    "pred_price": 0,
                    "title": "⚠️ 数据无效",
                    "desc": "数据包含无效值，无法预测",
                    "action": "建议：跳过该股票",
                    "color": "blue"
                }
                
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
            st.warning(f"AI预测失败: {str(e)}")
            return {
                "dates": ["明日", "后日", "大后日"],
                "prices": [0, 0, 0],
                "pred_price": 0,
                "title": "⚠️ 预测失败",
                "desc": "模型计算异常，无法生成预测",
                "action": "建议：忽略预测结果",
                "color": "blue"
            }

    def calc_indicators(self, df):
        """计算技术指标 - 异常加固"""
        if df is None or df.empty:
            return df
            
        try:
            df = df.copy()
            df['MA5'] = df['close'].rolling(5).mean()
            df['MA20'] = df['close'].rolling(20).mean()
            
            # MACD计算（容错）
            try:
                exp1 = df['close'].ewm(span=12, adjust=False).mean()
                exp2 = df['close'].ewm(span=26, adjust=False).mean()
                df['DIF'] = exp1 - exp2
                df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
                df['MACD'] = 2 * (df['DIF'] - df['DEA'])
            except:
                pass
                
            return df
        except Exception as e:
            st.warning(f"计算技术指标失败: {str(e)}")
            return df

    def plot_professional_kline(self, df, title):
        """绘制K线图 - 异常加固"""
        if df is None or df.empty or len(df) < 10:
            return None
            
        try:
            df = self.calc_indicators(df)
            
            # 买卖信号计算（容错）
            df['Signal'] = 0
            if 'MA5' in df.columns and 'MA20' in df.columns:
                try:
                    df.loc[
                        (df['MA5'] > df['MA20']) & (df['MA5'].shift(1) <= df['MA20'].shift(1)), 
                        'Signal'
                    ] = 1 
                    df.loc[
                        (df['MA5'] < df['MA20']) & (df['MA5'].shift(1) >= df['MA20'].shift(1)), 
                        'Signal'
                    ] = -1 
                except:
                    pass

            buy_points = df[df['Signal'] == 1]
            sell_points = df[df['Signal'] == -1]

            # 绘制K线图
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
            
            # 添加均线（容错）
            if 'MA5' in df.columns and not df['MA5'].isna().all():
                fig.add_trace(go.Scatter(
                    x=df['date'], 
                    y=df['MA5'], 
                    name='MA5', 
                    line=dict(color='orange', width=1)
                ))
            
            if 'MA20' in df.columns and not df['MA20'].isna().all():
                fig.add_trace(go.Scatter(
                    x=df['date'], 
                    y=df['MA20'], 
                    name='MA20', 
                    line=dict(color='blue', width=1)
                ))

            # 添加买卖点（容错）
            if not buy_points.empty:
                try:
                    fig.add_trace(go.Scatter(
                        x=buy_points['date'], 
                        y=buy_points['low']*0.98, 
                        mode='markers+text', 
                        marker=dict(symbol='triangle-up', size=12, color='red'), 
                        text='B', 
                        textposition='bottom center', 
                        name='买入'
                    ))
                except:
                    pass
            
            if not sell_points.empty:
                try:
                    fig.add_trace(go.Scatter(
                        x=sell_points['date'], 
                        y=sell_points['high']*1.02, 
                        mode='markers+text', 
                        marker=dict(symbol='triangle-down', size=12, color='green'), 
                        text='S', 
                        textposition='top center', 
                        name='卖出'
                    ))
                except:
                    pass

            fig.update_layout(
                title=f"{title} - 智能操盘K线", 
                xaxis_rangeslider_visible=False, 
                height=500,
                template="simple_white"
            )
            return fig
        except Exception as e:
            st.warning(f"绘制K线图失败: {str(e)}")
            return None

# ==========================================
# 3. 界面 UI
# ==========================================
def main():
    st.set_page_config(page_title="量化选股系统 V45", layout="wide")
    st.title("📈 智能量化选股系统 (V45)")
    st.markdown("---")

    # 初始化引擎
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

    # 侧边栏配置
    with st.sidebar:
        st.header("🕹️ 控制台")
        max_price_limit = st.slider("💰 价格上限 (元)", 3.0, 100.0, 20.0, step=1.0)
        pool_mode = st.radio("🔎 选股范围:", ("中证500 (中小盘)", "沪深300 (大盘)", "全市场扫描", "手动输入"))
        scan_limit = st.slider("🔢 扫描数量 (池大小)", 50, 6000, 500, step=50)

        # 股票池加载逻辑
        final_code_list = []
        if pool_mode == "手动输入":
            default_pool = "600519, 002131, 002312, 600580, 002594"
            target_pool_str = st.text_area("监控股票池", default_pool, height=100)
            final_code_list = [
                code.strip() for code in target_pool_str.replace("，", ",").split(",") 
                if code.strip()
            ]
        else:
            if st.button(f"📥 加载 {pool_mode} 成分股"):
                with st.spinner("正在获取成分股..."):
                    if pool_mode == "全市场扫描":
                        stock_list = engine.get_all_stocks()
                    elif "中证500" in pool_mode:
                        stock_list = engine.get_index_stocks("zz500")
                    else:
                        stock_list = engine.get_index_stocks("hs300")
                    
                    if stock_list:
                        st.session_state['full_pool'] = stock_list 
                        st.success(f"已加载全量 {len(stock_list)} 只股票")
                    else:
                        st.error("获取股票失败，请重试")
            
            if 'full_pool' in st.session_state and st.session_state['full_pool']:
                full_list = st.session_state['full_pool']
                final_code_list = full_list[:scan_limit] 
                st.info(f"池内待扫: {len(final_code_list)} 只 (总库: {len(full_list)})")
            else:
                st.info("请先点击上方按钮加载股票池")

        st.markdown("---")
        # 扫描触发按钮
        scan_trigger = st.button("🚀 启动全策略扫描 (V45)", type="primary")

    # 主界面逻辑
    if scan_trigger:
        if not final_code_list:
            st.sidebar.error("请先加载股票！")
        else:
            with st.spinner("📊 正在执行全策略扫描，请稍候..."):
                st.caption(f"当前筛选：价格 < {max_price_limit}元 | 剔除ST/科创/北交 | 模式：长连接稳定扫描")
                scan_res, alerts, valid_options = engine.scan_market_optimized(
                    final_code_list, 
                    max_price=max_price_limit
                )
                st.session_state['scan_res'] = scan_res
                st.session_state['valid_options'] = valid_options
                st.session_state['alerts'] = alerts

    # 扫描结果展示
    if st.session_state['scan_res']:
        st.subheader("🎯 扫描结果")
        
        # 分页展示（减轻前端压力）
        page_size = 20
        total_results = len(st.session_state['scan_res'])
        total_pages = (total_results + page_size - 1) // page_size
        
        col1, col2 = st.columns([1, 4])
        with col1:
            page = st.number_input("页码", min_value=1, max_value=total_pages, value=1)
        
        # 计算分页范围
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_results)
        paginated_results = st.session_state['scan_res'][start_idx:end_idx]
        
        # 展示表格
        if paginated_results:
            df_res = pd.DataFrame(paginated_results)
            # 调整列顺序，隐藏priority列
            df_res = df_res[[
                "代码", "名称", "所属行业", "现价", "涨跌", 
                "获利筹码", "风险评级", "策略信号", "综合评级"
            ]]
            st.dataframe(df_res, use_container_width=True)
            
            # 高亮高优先级股票
            if st.session_state['alerts']:
                st.markdown("### ⚡ 高优先级预警")
                alert_text = " | ".join(st.session_state['alerts'])
                st.markdown(f"<span style='color:red; font-size:18px;'>{alert_text}</span>", unsafe_allow_html=True)
        
        # 结果统计
        st.info(f"本次扫描共命中 {total_results} 只符合条件的股票 (共扫描 {len(final_code_list)} 只)")

    # 策略说明
    with st.expander("📋 策略说明", expanded=False):
        st.write("### 核心选股策略说明")
        for key, desc in STRATEGY_DESC.items():
            st.write(f"- {key}: {desc}")

if __name__ == "__main__":
    main()
