import streamlit as st
import hashlib

# ==========================================
# 🔐 密码保护模块（增强版 - 使用Secrets）
# ==========================================

# ==========================================
# ⚠️ 核心配置（必须在最前面，在任何其他streamlit命令之前）
# ==========================================
st.set_page_config(
    page_title="V45 完美说明书版", 
    layout="wide", 
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

def get_password():
    """从Secrets获取密码，如果没有则使用默认值"""
    try:
        # 尝试从Streamlit Secrets获取密码
        if hasattr(st, 'secrets') and st.secrets is not None:
            password = st.secrets.get("PASSWORD", "vip666888")
        else:
            password = "vip666888"
    except Exception:
        # 如果Secrets不存在（本地运行），使用默认值
        password = "vip666888"
    return password

# 获取密码并计算哈希值
PASSWORD = get_password()
PASSWORD_HASH = hashlib.sha256(PASSWORD.encode()).hexdigest()

def check_password():
    """密码验证函数（增强版）"""
    if 'password_correct' not in st.session_state:
        st.session_state.password_correct = False
    
    # 限制登录尝试次数
    if 'login_attempts' not in st.session_state:
        st.session_state.login_attempts = 0
    
    # 如果尝试次数过多，阻止访问
    if st.session_state.login_attempts >= 5:
        st.error("❌ 登录尝试次数过多，请稍后再试或联系管理员")
        st.info("💡 提示：如果忘记密码，请联系系统管理员")
        st.stop()
    
    if not st.session_state.password_correct:
        st.title("🔐 系统访问验证")
        st.markdown("---")
        st.info("💡 请输入访问密码以继续使用系统")
        
        password_input = st.text_input("请输入访问密码:", type="password", key="pwd_input")
        
        if st.button("🔓 验证", type="primary"):
            # 使用哈希验证（更安全）
            input_hash = hashlib.sha256(password_input.encode()).hexdigest()
            if input_hash == PASSWORD_HASH:
                st.session_state.password_correct = True
                st.session_state.login_attempts = 0
                st.success("✅ 验证成功！")
                st.rerun()
            else:
                st.session_state.login_attempts += 1
                remaining = 5 - st.session_state.login_attempts
                if remaining > 0:
                    st.error(f"❌ 密码错误，请重试！（剩余尝试次数：{remaining}）")
                else:
                    st.error("❌ 登录尝试次数已达上限，请稍后再试")
                st.stop()
        else:
            st.stop()
    
    return True

# 执行密码验证
if not check_password():
    st.stop()

# ==========================================
# 🆕 V45 新功能提示
# ==========================================
st.markdown("""
<div style='background-color: #00ff00; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
<h2 style='color: #000; text-align: center;'>🎉 V45 新功能已激活！🎉</h2>
<p style='color: #000; font-size: 16px; text-align: center;'>
✅ 多策略并行扫描 | ✅ 策略参数自定义 | ✅ 历史回测功能 | ✅ 结果导出 | ✅ 扫描历史记录
</p>
<p style='color: #000; font-size: 14px; text-align: center;'>
👉 请查看左侧边栏的"🆕 V45 新功能"区域
</p>
</div>
""", unsafe_allow_html=True)

st.title("🛡️ V45 智能量化系统 (全信号图例版)")
st.caption("✅ 系统已就绪 | 核心组件加载完成 | 支持6000股扫描 | V45 Build")

# 新功能高亮显示
st.success("🎉 **V45 新功能已激活！** 左侧边栏现在支持多策略扫描、参数自定义和历史记录管理")

with st.expander("💡 查看V45新功能详情", expanded=True):
    col_new1, col_new2 = st.columns(2)
    
    with col_new1:
        st.markdown("#### 🆕 主要新功能")
        st.markdown("- 🔀 **多策略并行扫描**: 同时运行9种不同策略")
        st.markdown("- 📊 **策略对比模式**: 分析不同策略效果")
        st.markdown("- ⚙️ **参数自定义**: 调整技术指标参数")
        st.markdown("- 📈 **历史回测**: 验证策略有效性")
    
    with col_new2:
        st.markdown("#### 🎯 使用方式")
        st.markdown("1. **左侧边栏** - 选择多个扫描策略")
        st.markdown("2. **高级参数** - 展开设置技术指标")
        st.markdown("3. **扫描结果** - 查看导出和回测功能")
        st.markdown("4. **历史记录** - 管理过往扫描结果")
    
    st.warning("⚠️ **重要**: 如果看不到新功能，请刷新页面或清除浏览器缓存！")

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
    st.error(f"请检查 requirements.txt 文件是否已上传到GitHub")
    st.stop()
except Exception as e:
    st.error(f"❌ 导入错误: {e}")
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
💎 RSI超卖反弹: RSI<30后回升，超跌反弹机会。
📊 布林带突破: 价格突破布林带上轨，强势突破信号。
🎯 KDJ金叉: K线上穿D线，短期买入信号。
📉 200日均线趋势: 价格站上200日均线，长期上升趋势。
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
    "📈 多头排列": "昨日收阳 且 今日收盘价 > 昨日收盘价",
    "💎 RSI超卖反弹": "RSI<30后回升至35以上，超跌反弹机会",
    "📊 布林带突破": "价格突破布林带上轨 + 成交量放大",
    "🎯 KDJ金叉": "K线上穿D线 + RSI>50，短期买入信号",
    "📉 200日均线趋势": "价格站上200日均线 + 均线向上，长期上升趋势"
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
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # 先尝试登出（如果之前有登录）
                try:
                    bs.logout()
                except:
                    pass
                
                # 尝试登录
                login_result = bs.login()
                if login_result.error_code != '0':
                    last_error = f"登录失败: {login_result.error_msg if hasattr(login_result, 'error_msg') else '未知错误'}"
                    if attempt < max_retries - 1:
                        time.sleep(2)  # 等待2秒后重试
                        continue
                    return []
                
                # 查询所有股票
                rs = bs.query_all_stock()
                if rs.error_code != '0':
                    last_error = f"查询失败: {rs.error_msg if hasattr(rs, 'error_msg') else '未知错误'}"
                    bs.logout()
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    return []
                
                stocks = []
                data_list = []
                count = 0
                max_count = 10000  # 防止无限循环
                
                # 修复：使用 and 而不是 &，并正确处理 rs.next() 的返回值
                while rs.error_code == '0' and count < max_count:
                    if not rs.next():
                        break
                    row_data = rs.get_row_data()
                    if row_data and len(row_data) >= 2:
                        data_list.append(row_data)
                    count += 1
                
                if not data_list:
                    last_error = "未获取到任何股票数据"
                    bs.logout()
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    return []
                
                for data in data_list:
                    if len(data) >= 2:
                        code = data[0]
                        name = data[1] if len(data) > 1 else ""
                        if self.is_valid(code, name):
                            stocks.append(code)
                
                bs.logout()
                
                if stocks:
                    return stocks[:self.MAX_SCAN_LIMIT]
                else:
                    last_error = "过滤后没有有效股票"
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    return []
                    
            except Exception as e:
                last_error = f"异常错误: {str(e)}"
                try:
                    bs.logout()
                except:
                    pass
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                return []
        
        return []

    def get_index_stocks(self, index_type="zz500"):
        """获取指数成分股"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # 尝试登录
                login_result = bs.login()
                if login_result.error_code != '0':
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                    return []
                
                stocks = []
                try:
                    # 先尝试登出（如果之前有登录）
                    try:
                        bs.logout()
                    except:
                        pass
                    
                    # 尝试登录
                    login_result = bs.login()
                    if login_result.error_code != '0':
                        if attempt < max_retries - 1:
                            time.sleep(2)
                            continue
                        return []
                    
                    if index_type == "hs300": 
                        rs = bs.query_hs300_stocks()
                    else: 
                        rs = bs.query_zz500_stocks()
                    
                    if rs.error_code != '0':
                        bs.logout()
                        if attempt < max_retries - 1:
                            time.sleep(2)
                            continue
                        return []
                    
                    # 安全获取数据
                    while rs.next(): 
                        row_data = rs.get_row_data()
                        if row_data and len(row_data) >= 2:
                            code = row_data[1]  # 股票代码在第二个位置
                            if code and code.strip():
                                stocks.append(code)
                except Exception as e:
                    try:
                        bs.logout()
                    except:
                        pass
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    return []
                finally: 
                    try:
                        bs.logout()
                    except:
                        pass
                
                if stocks:
                    return stocks[:self.MAX_SCAN_LIMIT]
                else:
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                    return []
                    
            except Exception as e:
                try:
                    bs.logout()
                except:
                    pass
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                return []
        
        return []

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
    
    def calc_rsi(self, df, period=14):
        """计算RSI相对强弱指标"""
        try:
            if len(df) < period + 1:
                return None
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else None
        except:
            return None
    
    def calc_kdj(self, df, period=9):
        """计算KDJ指标"""
        try:
            if len(df) < period + 1:
                return None, None, None
            low_min = df['low'].rolling(window=period).min()
            high_max = df['high'].rolling(window=period).max()
            rsv = (df['close'] - low_min) / (high_max - low_min) * 100
            
            k = rsv.ewm(com=2, adjust=False).mean()
            d = k.ewm(com=2, adjust=False).mean()
            j = 3 * k - 2 * d
            
            return k.iloc[-1], d.iloc[-1], j.iloc[-1]
        except:
            return None, None, None
    
    def calc_bollinger(self, df, period=20, std_dev=2):
        """计算布林带指标"""
        try:
            if len(df) < period:
                return None, None, None
            ma = df['close'].rolling(window=period).mean()
            std = df['close'].rolling(window=period).std()
            upper = ma + (std * std_dev)
            lower = ma - (std * std_dev)
            return upper.iloc[-1], ma.iloc[-1], lower.iloc[-1]
        except:
            return None, None, None

    def _process_single_stock(self, code, max_price=None, strategy_params=None):
        # 设置默认参数
        if strategy_params is None:
            strategy_params = {
                'ma5_period': 5, 'ma20_period': 20, 'ma200_period': 200,
                'bb_period': 20, 'rsi_oversold': 30, 'rsi_overbought': 70,
                'kdj_period': 9, 'vol_ratio_threshold': 1.8,
                'win_rate_threshold': 70, 'limit_up_count': 3,
                'turnover_threshold': 5, 'gentle_rise_threshold': 5
            }
        
        code = self.clean_code(code)
        end = datetime.datetime.now().strftime("%Y-%m-%d")
        start = (datetime.datetime.now() - datetime.timedelta(days=200)).strftime("%Y-%m-%d")  # 增加数据长度以支持更长周期均线
        
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
        except Exception as e:
            return None

        if not data: return None
        try:
            df = pd.DataFrame(data, columns=["date", "open", "close", "high", "low", "volume", "pctChg", "turn"])
            df = df.apply(pd.to_numeric, errors='coerce')
        except Exception as e:
            return None
        if len(df) < 60: return None

        curr = df.iloc[-1]
        prev = df.iloc[-2]
        if max_price is not None:
            if curr['close'] > max_price: return None

        winner_rate = self.calc_winner_rate(df, curr['close'])
        try: ipo_date = datetime.datetime.strptime(info['ipoDate'], "%Y-%m-%d")
        except: ipo_date = datetime.datetime(2000, 1, 1)
        days_listed = (datetime.datetime.now() - ipo_date).days

        # 使用动态参数计算均线
        df['MA5'] = df['close'].rolling(strategy_params['ma5_period']).mean()
        df['MA20'] = df['close'].rolling(strategy_params['ma20_period']).mean()
        if len(df) >= strategy_params['ma200_period']:
            df['MA200'] = df['close'].rolling(strategy_params['ma200_period']).mean()
        else:
            df['MA200'] = pd.Series([None] * len(df))
        risk_level = self.calc_risk_level(curr['close'], df['MA5'].iloc[-1], df['MA20'].iloc[-1])

        # 使用动态参数计算技术指标
        rsi = self.calc_rsi(df, period=14)  # RSI周期保持14天
        k, d, j = self.calc_kdj(df, period=strategy_params['kdj_period'])
        bb_upper, bb_mid, bb_lower = self.calc_bollinger(df, period=strategy_params['bb_period'])

        signal_tags = []
        priority = 0
        action = "WAIT (观望)"

        is_3_up = all(df['pctChg'].tail(3) > 0)
        sum_3_rise = df['pctChg'].tail(3).sum()
        if (is_3_up and sum_3_rise <= strategy_params['gentle_rise_threshold'] and winner_rate > 62):
            signal_tags.append("🔴温和吸筹")
            priority = max(priority, 60)
            action = "BUY (低吸)"

        is_high_turn = all(df['turn'].tail(2) > strategy_params['turnover_threshold']) 
        if is_high_turn and winner_rate > strategy_params['win_rate_threshold']:
            signal_tags.append("🔥换手锁仓")
            priority = max(priority, 70)
            action = "BUY (博弈)"

        df_60 = df.tail(60)
        limit_up_60 = len(df_60[df_60['pctChg'] > 9.5])
        if limit_up_60 >= strategy_params['limit_up_count'] and winner_rate > 80 and days_listed > 30:
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
        is_double_vol = (curr['volume'] > prev['volume'] * strategy_params['vol_ratio_threshold']) or (curr['volume'] > vol_ma5 * strategy_params['vol_ratio_threshold'])

        if has_limit_up_20 and has_gap and has_streak and is_double_vol:
            signal_tags.append("👑四星共振")
            priority = 100
            action = "STRONG BUY"
        elif prev['open'] < prev['close'] and curr['close'] > prev['close']: 
             if priority == 0: 
                 action = "HOLD (持有)"
                 priority = 10
                 signal_tags.append("📈多头")
        
        # 新增策略：RSI超卖反弹（使用动态参数）
        if rsi is not None:
            if rsi < strategy_params['rsi_oversold'] and len(df) >= 2:
                prev_rsi = self.calc_rsi(df.iloc[:-1])
                if prev_rsi is not None and prev_rsi < rsi and rsi > 35:
                    signal_tags.append("💎RSI超卖反弹")
                    priority = max(priority, 65)
                    if action == "WAIT (观望)":
                        action = "BUY (低吸)"
        
        # 新增策略：布林带突破
        if bb_upper is not None and bb_lower is not None:
            if curr['close'] > bb_upper and curr['volume'] > df['volume'].tail(20).mean() * 1.2:
                signal_tags.append("📊布林带突破")
                priority = max(priority, 75)
                if action in ["WAIT (观望)", "HOLD (持有)"]:
                    action = "BUY (博弈)"
        
        # 新增策略：KDJ金叉
        if k is not None and d is not None:
            if len(df) >= 2:
                prev_k, prev_d, _ = self.calc_kdj(df.iloc[:-1], period=strategy_params['kdj_period'])
                if prev_k is not None and prev_d is not None:
                    if prev_k <= prev_d and k > d and rsi is not None and rsi > 50:
                        signal_tags.append("🎯KDJ金叉")
                        priority = max(priority, 70)
                        if action in ["WAIT (观望)", "HOLD (持有)"]:
                            action = "BUY (博弈)"
        
        # 新增策略：均线趋势（使用动态参数）
        if len(df) >= strategy_params['ma200_period'] and not pd.isna(df['MA200'].iloc[-1]):
            ma200_current = df['MA200'].iloc[-1]
            ma200_prev = df['MA200'].iloc[-2] if len(df) >= strategy_params['ma200_period'] + 1 else ma200_current
            if curr['close'] > ma200_current and ma200_current > ma200_prev:
                signal_tags.append("📉长期均线趋势")
                priority = max(priority, 80)
                if action in ["WAIT (观望)", "HOLD (持有)", "BUY (低吸)"]:
                    action = "BUY (低吸)" if action == "WAIT (观望)" else action

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

    def _process_multi_strategy_stock(self, code, max_price=None, selected_strategies=None, strategy_params=None):
        """多策略处理单只股票"""
        # 设置默认参数
        if strategy_params is None:
            strategy_params = {
                'ma5_period': 5, 'ma20_period': 20, 'ma200_period': 200,
                'bb_period': 20, 'rsi_oversold': 30, 'rsi_overbought': 70,
                'kdj_period': 9, 'vol_ratio_threshold': 1.8,
                'win_rate_threshold': 70, 'limit_up_count': 3,
                'turnover_threshold': 5, 'gentle_rise_threshold': 5
            }
        
        if selected_strategies is None:
            selected_strategies = ["均线突破", "RSI超卖反弹", "KDJ金叉"]
        
        code = self.clean_code(code)
        end = datetime.datetime.now().strftime("%Y-%m-%d")
        start = (datetime.datetime.now() - datetime.timedelta(days=200)).strftime("%Y-%m-%d")
        
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
        except Exception as e:
            return None

        if not data: return None
        try:
            df = pd.DataFrame(data, columns=["date", "open", "close", "high", "low", "volume", "pctChg", "turn"])
            df = df.apply(pd.to_numeric, errors='coerce')
        except Exception as e:
            return None
        if len(df) < 60: return None

        curr = df.iloc[-1]
        prev = df.iloc[-2]
        if max_price is not None:
            if curr['close'] > max_price: return None

        winner_rate = self.calc_winner_rate(df, curr['close'])
        try: ipo_date = datetime.datetime.strptime(info['ipoDate'], "%Y-%m-%d")
        except: ipo_date = datetime.datetime(2000, 1, 1)
        days_listed = (datetime.datetime.now() - ipo_date).days

        # 使用动态参数计算均线
        df['MA5'] = df['close'].rolling(strategy_params['ma5_period']).mean()
        df['MA20'] = df['close'].rolling(strategy_params['ma20_period']).mean()
        if len(df) >= strategy_params['ma200_period']:
            df['MA200'] = df['close'].rolling(strategy_params['ma200_period']).mean()
        else:
            df['MA200'] = pd.Series([None] * len(df))
        risk_level = self.calc_risk_level(curr['close'], df['MA5'].iloc[-1], df['MA20'].iloc[-1])

        # 使用动态参数计算技术指标
        rsi = self.calc_rsi(df, period=14)
        k, d, j = self.calc_kdj(df, period=strategy_params['kdj_period'])
        bb_upper, bb_mid, bb_lower = self.calc_bollinger(df, period=strategy_params['bb_period'])

        # 多策略信号检测
        strategy_results = {}
        
        # 1. 均线突破策略
        if "均线突破" in selected_strategies:
            ma_signals = []
            ma_priority = 0
            ma_action = "WAIT"
            
            if prev['open'] < prev['close'] and curr['close'] > prev['close']:
                ma_signals.append("📈多头")
                ma_priority = 10
                ma_action = "HOLD"
            
            # MA金叉
            if (not pd.isna(df['MA5'].iloc[-1]) and not pd.isna(df['MA20'].iloc[-1]) and
                not pd.isna(df['MA5'].iloc[-2]) and not pd.isna(df['MA20'].iloc[-2])):
                if df['MA5'].iloc[-2] <= df['MA20'].iloc[-2] and df['MA5'].iloc[-1] > df['MA20'].iloc[-1]:
                    ma_signals.append("🚀金叉突变")
                    ma_priority = max(ma_priority, 75)
                    ma_action = "BUY (博弈)"
                    
            strategy_results["均线突破"] = {
                'signals': ma_signals,
                'priority': ma_priority,
                'action': ma_action
            }
        
        # 2. RSI超卖反弹策略
        if "RSI超卖反弹" in selected_strategies:
            rsi_signals = []
            rsi_priority = 0
            rsi_action = "WAIT"
            
            if rsi is not None:
                if rsi < strategy_params['rsi_oversold'] and len(df) >= 2:
                    prev_rsi = self.calc_rsi(df.iloc[:-1])
                    if prev_rsi is not None and prev_rsi < rsi and rsi > 35:
                        rsi_signals.append("💎RSI超卖反弹")
                        rsi_priority = 65
                        rsi_action = "BUY (低吸)"
                        
            strategy_results["RSI超卖反弹"] = {
                'signals': rsi_signals,
                'priority': rsi_priority,
                'action': rsi_action
            }
        
        # 3. KDJ金叉策略
        if "KDJ金叉" in selected_strategies:
            kdj_signals = []
            kdj_priority = 0
            kdj_action = "WAIT"
            
            if k is not None and d is not None and len(df) >= 2:
                prev_k, prev_d, _ = self.calc_kdj(df.iloc[:-1], period=strategy_params['kdj_period'])
                if prev_k is not None and prev_d is not None:
                    if prev_k <= prev_d and k > d and rsi is not None and rsi > 50:
                        kdj_signals.append("🎯KDJ金叉")
                        kdj_priority = 70
                        kdj_action = "BUY (博弈)"
                        
            strategy_results["KDJ金叉"] = {
                'signals': kdj_signals,
                'priority': kdj_priority,
                'action': kdj_action
            }
        
        # 4. 布林带突破策略
        if "布林带突破" in selected_strategies:
            bb_signals = []
            bb_priority = 0
            bb_action = "WAIT"
            
            if bb_upper is not None and bb_lower is not None:
                if curr['close'] > bb_upper and curr['volume'] > df['volume'].tail(20).mean() * 1.2:
                    bb_signals.append("📊布林带突破")
                    bb_priority = 75
                    bb_action = "BUY (博弈)"
                    
            strategy_results["布林带突破"] = {
                'signals': bb_signals,
                'priority': bb_priority,
                'action': bb_action
            }
        
        # 5. 温和吸筹策略
        if "温和吸筹" in selected_strategies:
            gentle_signals = []
            gentle_priority = 0
            gentle_action = "WAIT"
            
            is_3_up = all(df['pctChg'].tail(3) > 0)
            sum_3_rise = df['pctChg'].tail(3).sum()
            if (is_3_up and sum_3_rise <= strategy_params['gentle_rise_threshold'] and winner_rate > 62):
                gentle_signals.append("🔴温和吸筹")
                gentle_priority = 60
                gentle_action = "BUY (低吸)"
                
            strategy_results["温和吸筹"] = {
                'signals': gentle_signals,
                'priority': gentle_priority,
                'action': gentle_action
            }
        
        # 6. 换手锁仓策略
        if "换手锁仓" in selected_strategies:
            turnover_signals = []
            turnover_priority = 0
            turnover_action = "WAIT"
            
            is_high_turn = all(df['turn'].tail(2) > strategy_params['turnover_threshold'])
            if is_high_turn and winner_rate > strategy_params['win_rate_threshold']:
                turnover_signals.append("🔥换手锁仓")
                turnover_priority = 70
                turnover_action = "BUY (博弈)"
                
            strategy_results["换手锁仓"] = {
                'signals': turnover_signals,
                'priority': turnover_priority,
                'action': turnover_action
            }
        
        # 7. 妖股基因策略
        if "妖股基因" in selected_strategies:
            demon_signals = []
            demon_priority = 0
            demon_action = "WAIT"
            
            df_60 = df.tail(60)
            limit_up_60 = len(df_60[df_60['pctChg'] > 9.5])
            if limit_up_60 >= strategy_params['limit_up_count'] and winner_rate > 80 and days_listed > 30:
                demon_signals.append("🐲妖股基因")
                demon_priority = 90
                demon_action = "STRONG BUY"
                
            strategy_results["妖股基因"] = {
                'signals': demon_signals,
                'priority': demon_priority,
                'action': demon_action
            }
        
        # 8. 四星共振策略
        if "四星共振" in selected_strategies:
            resonance_signals = []
            resonance_priority = 0
            resonance_action = "WAIT"
            
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
            is_double_vol = (curr['volume'] > prev['volume'] * strategy_params['vol_ratio_threshold']) or (curr['volume'] > vol_ma5 * strategy_params['vol_ratio_threshold'])

            if has_limit_up_20 and has_gap and has_streak and is_double_vol:
                resonance_signals.append("👑四星共振")
                resonance_priority = 100
                resonance_action = "STRONG BUY"
                
            strategy_results["四星共振"] = {
                'signals': resonance_signals,
                'priority': resonance_priority,
                'action': resonance_action
            }
        
        # 汇总所有策略结果
        all_signals = []
        max_priority = 0
        best_action = "WAIT (观望)"
        
        for strategy_name, result in strategy_results.items():
            if result['signals']:
                all_signals.extend(result['signals'])
            if result['priority'] > max_priority:
                max_priority = result['priority']
                best_action = result['action']
        
        if max_priority == 0:
            return None
        
        return {
            "result": {
                "代码": code, "名称": info['name'], 
                "所属行业": info['industry'],
                "现价": curr['close'], 
                "涨跌": f"{curr['pctChg']:.2f}%", 
                "获利筹码": winner_rate,
                "风险评级": risk_level,
                "策略信号": " + ".join(all_signals),
                "综合评级": best_action,
                "priority": max_priority
            },
            "alert": f"{info['name']}" if max_priority >= 90 else None,
            "option": f"{code} | {info['name']}",
            "strategy_breakdown": strategy_results
        }

    def scan_market_optimized(self, code_list, max_price=None, strategy_params=None):
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
                res = self._process_single_stock(code, max_price, strategy_params)
                if res:
                    results.append(res["result"])
                    if res["alert"]: alerts.append(res["alert"])
                    valid_codes_list.append(res["option"])
            except Exception as e:
                try:
                    bs.logout()
                    time.sleep(0.5)
                    bs.login()
                except:
                    pass
                continue

        bs.logout()
        progress_container.empty()
        
        # 显示扫描完成提示
        if len(results) > 0:
            st.success(f"✅ 扫描完成！共找到 {len(results)} 只符合条件的股票")
        else:
            st.info(f"ℹ️ 扫描完成！共扫描 {total} 只股票，未找到符合条件的股票")
        
        return results, alerts, valid_codes_list

    def scan_market_multi_strategy(self, code_list, max_price=None, selected_strategies=None, strategy_params=None, comparison_mode=False):
        """多策略并行扫描市场"""
        results, alerts, valid_codes_list = [], [], []
        strategy_stats = {}  # 策略统计
        
        lg = bs.login()
        if lg.error_code != '0':
            st.error("连接服务器失败，请检查网络！")
            return [], [], []

        if len(code_list) > self.MAX_SCAN_LIMIT:
            code_list = code_list[:self.MAX_SCAN_LIMIT]
            st.info(f"⚠️ 股票数量超过限制，已截取前{self.MAX_SCAN_LIMIT}只")

        total = len(code_list)
        
        progress_container = st.empty()
        progress_bar = progress_container.progress(0, text=f"🚀 正在启动多策略扫描 (共 {total} 只)...")
        
        BATCH_SIZE = 20
        
        # 初始化策略统计
        if selected_strategies:
            for strategy in selected_strategies:
                strategy_stats[strategy] = {'count': 0, 'stocks': []}
        
        for i, code in enumerate(code_list):
            if i % BATCH_SIZE == 0 or i == total - 1:
                progress = (i + 1) / total
                current_count = min(i + 1, total)
                strategy_summary = " | ".join([f"{s}:{strategy_stats[s]['count']}" for s in strategy_stats.keys()][:3])
                progress_bar.progress(progress, 
                                    text=f"🔍 多策略分析: {code} ({current_count}/{total}) | {strategy_summary}")
            
            try:
                res = self._process_multi_strategy_stock(code, max_price, selected_strategies, strategy_params)
                if res:
                    results.append(res["result"])
                    if res["alert"]: alerts.append(res["alert"])
                    valid_codes_list.append(res["option"])
                    
                    # 统计各策略命中情况
                    if "strategy_breakdown" in res:
                        for strategy_name, strategy_result in res["strategy_breakdown"].items():
                            if strategy_result['signals']:
                                strategy_stats[strategy_name]['count'] += 1
                                strategy_stats[strategy_name]['stocks'].append({
                                    'code': res["result"]["代码"],
                                    'name': res["result"]["名称"],
                                    'signals': strategy_result['signals'],
                                    'priority': strategy_result['priority'],
                                    'action': strategy_result['action']
                                })
                    
            except Exception as e:
                try:
                    bs.logout()
                    time.sleep(0.5)
                    bs.login()
                except:
                    pass
                continue

        bs.logout()
        progress_container.empty()
        
        # 保存策略统计到session_state
        st.session_state['strategy_stats'] = strategy_stats
        
        # 显示扫描完成提示和策略统计
        if len(results) > 0:
            st.success(f"✅ 多策略扫描完成！共找到 {len(results)} 只符合条件的股票")
            
            # 显示策略统计
            stats_text = " | ".join([f"{strategy}: {stats['count']}只" for strategy, stats in strategy_stats.items() if stats['count'] > 0])
            if stats_text:
                st.info(f"📊 策略命中统计：{stats_text}")
        else:
            st.info(f"ℹ️ 多策略扫描完成！共扫描 {total} 只股票，未找到符合条件的股票")
        
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
            except Exception:
                pass
            return None

    def run_ai_prediction(self, df):
        """AI预测 - 改进版，使用更多特征"""
        if df is None or len(df) < 30:
            return None
            
        try:
            # 使用更多历史数据
            recent = df.tail(30).reset_index(drop=True)
            
            # 计算特征：价格、成交量、技术指标
            X_features = []
            y_values = []
            
            for i in range(5, len(recent)):
                features = [
                    recent.iloc[i-1]['close'],
                    recent.iloc[i-2]['close'] if i >= 2 else recent.iloc[i-1]['close'],
                    recent.iloc[i-1]['volume'],
                    recent.iloc[i-1]['close'] - recent.iloc[i-2]['close'] if i >= 2 else 0,
                ]
                # 添加移动平均特征
                if i >= 5:
                    features.append(recent.iloc[i-5:i]['close'].mean())
                else:
                    features.append(recent.iloc[i-1]['close'])
                
                X_features.append(features)
                y_values.append(recent.iloc[i]['close'])
            
            if len(X_features) < 5:
                return None
            
            X = np.array(X_features)
            y = np.array(y_values)
            
            # 检查数据有效性
            if np.isnan(X).any() or np.isnan(y).any():
                return None
                
            model = LinearRegression()
            model.fit(X, y)
            
            # 预测未来3天
            last_features = X_features[-1]
            pred_prices = []
            for day in range(1, 4):
                # 使用前一天的预测作为输入（简化版）
                if day == 1:
                    pred_price = model.predict([last_features])[0]
                else:
                    # 更新特征进行预测
                    new_features = last_features.copy()
                    new_features[0] = pred_prices[-1]  # 使用前一天的预测
                    pred_price = model.predict([new_features])[0]
                pred_prices.append(max(0, pred_price))  # 确保价格不为负
            
            future_dates = []
            current_date = datetime.date.today()
            for i in range(1, 4):
                d = current_date + datetime.timedelta(days=i)
                future_dates.append(d.strftime("%Y-%m-%d"))

            # 计算趋势斜率（基于预测价格的变化）
            slope = (pred_prices[1] - pred_prices[0]) / pred_prices[0] if pred_prices[0] > 0 else 0
            last_price = df['close'].iloc[-1]
            
            # 基于预测价格变化率判断趋势
            price_change_pct = (pred_prices[1] - last_price) / last_price * 100 if last_price > 0 else 0
            
            if price_change_pct > 2:
                hint_title = "🚀 上升通道加速中"
                hint_desc = f"惯性推演：股价将在 **{future_dates[1]}** 尝试冲击 **¥{pred_prices[1]:.2f}** (预计涨幅 {price_change_pct:.2f}%)。"
                action = "建议：坚定持有 / 逢低买入"
                color = "red"
            elif price_change_pct > 0:
                hint_title = "📈 震荡缓慢上行"
                hint_desc = f"趋势温和，预计 **{future_dates[1]}** 到达 **¥{pred_prices[1]:.2f}** (预计涨幅 {price_change_pct:.2f}%)。"
                action = "建议：耐心持股"
                color = "red"
            elif price_change_pct < -2:
                hint_title = "📉 下跌趋势加速"
                hint_desc = f"空头较强，预计 **{future_dates[1]}** 回落至 **¥{pred_prices[1]:.2f}** (预计跌幅 {abs(price_change_pct):.2f}%)。"
                action = "建议：反弹卖出"
                color = "green"
            else:
                hint_title = "⚖️ 横盘震荡"
                hint_desc = f"多空平衡，预计 **{future_dates[1]}** 在 **¥{pred_prices[1]:.2f}** 震荡 (预计变化 {price_change_pct:.2f}%)。"
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
        """计算技术指标 - 增加异常处理，包含RSI、KDJ、布林带等"""
        if df is None or df.empty:
            return df
            
        try:
            df = df.copy()
            df['MA5'] = df['close'].rolling(5).mean()
            df['MA20'] = df['close'].rolling(20).mean()
            if len(df) >= 200:
                df['MA200'] = df['close'].rolling(200).mean()
            
            # 计算MACD
            try:
                exp1 = df['close'].ewm(span=12, adjust=False).mean()
                exp2 = df['close'].ewm(span=26, adjust=False).mean()
                df['DIF'] = exp1 - exp2
                df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
                df['MACD'] = 2 * (df['DIF'] - df['DEA'])
            except Exception:
                pass
            
            # 计算RSI
            try:
                if len(df) >= 15:
                    delta = df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    df['RSI'] = 100 - (100 / (1 + rs))
            except Exception:
                pass
            
            # 计算KDJ
            try:
                if len(df) >= 10:
                    period = 9
                    low_min = df['low'].rolling(window=period).min()
                    high_max = df['high'].rolling(window=period).max()
                    rsv = (df['close'] - low_min) / (high_max - low_min) * 100
                    df['K'] = rsv.ewm(com=2, adjust=False).mean()
                    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
                    df['J'] = 3 * df['K'] - 2 * df['D']
            except Exception:
                pass
            
            # 计算布林带
            try:
                if len(df) >= 20:
                    period = 20
                    std_dev = 2
                    df['BB_Mid'] = df['close'].rolling(window=period).mean()
                    std = df['close'].rolling(window=period).std()
                    df['BB_Upper'] = df['BB_Mid'] + (std * std_dev)
                    df['BB_Lower'] = df['BB_Mid'] - (std * std_dev)
            except Exception:
                pass
                
            return df
        except Exception:
            return df

    def run_backtest(self, df, strategy_params=None):
        """运行单标的回测"""
        if df is None or len(df) < 30:
            return None
        
        # 设置默认参数
        if strategy_params is None:
            strategy_params = {
                'ma5_period': 5, 'ma20_period': 20, 'ma200_period': 200,
                'bb_period': 20, 'rsi_oversold': 30, 'rsi_overbought': 70,
                'kdj_period': 9, 'vol_ratio_threshold': 1.8,
                'win_rate_threshold': 70, 'limit_up_count': 3,
                'turnover_threshold': 5, 'gentle_rise_threshold': 5
            }
        
        try:
            # 计算技术指标
            df_backtest = self.calc_indicators(df.copy())
            
            # 生成交易信号
            df_backtest['buy_signal'] = 0
            df_backtest['sell_signal'] = 0
            df_backtest['position'] = 0  # 持仓状态：0=空仓，1=持仓
            df_backtest['returns'] = 0.0  # 每日收益率
            df_backtest['cumulative_returns'] = 1.0  # 累积收益率
            
            # 计算信号
            for i in range(max(strategy_params['ma20_period'], strategy_params['kdj_period']), len(df_backtest)):
                current_row = df_backtest.iloc[i]
                prev_row = df_backtest.iloc[i-1] if i > 0 else current_row
                
                # MA金叉买入信号
                if (current_row['MA5'] > current_row['MA20'] and 
                    prev_row['MA5'] <= prev_row['MA20'] and 
                    not pd.isna(current_row['MA5']) and not pd.isna(current_row['MA20'])):
                    df_backtest.iloc[i, df_backtest.columns.get_loc('buy_signal')] = 1
                
                # MA死叉卖出信号
                if (current_row['MA5'] < current_row['MA20'] and 
                    prev_row['MA5'] >= prev_row['MA20'] and 
                    not pd.isna(current_row['MA5']) and not pd.isna(current_row['MA20'])):
                    df_backtest.iloc[i, df_backtest.columns.get_loc('sell_signal')] = 1
                
                # RSI超卖买入信号
                if ('RSI' in df_backtest.columns and not pd.isna(current_row['RSI']) and 
                    current_row['RSI'] < strategy_params['rsi_oversold'] and 
                    i > 0 and not pd.isna(df_backtest.iloc[i-1]['RSI']) and
                    df_backtest.iloc[i-1]['RSI'] < current_row['RSI']):
                    df_backtest.iloc[i, df_backtest.columns.get_loc('buy_signal')] = 1
                
                # KDJ金叉买入信号
                if ('K' in df_backtest.columns and 'D' in df_backtest.columns and 
                    not pd.isna(current_row['K']) and not pd.isna(current_row['D']) and
                    current_row['K'] > current_row['D'] and 
                    i > 0 and not pd.isna(df_backtest.iloc[i-1]['K']) and not pd.isna(df_backtest.iloc[i-1]['D']) and
                    df_backtest.iloc[i-1]['K'] <= df_backtest.iloc[i-1]['D']):
                    df_backtest.iloc[i, df_backtest.columns.get_loc('buy_signal')] = 1
            
            # 执行交易逻辑
            position = 0
            buy_price = 0
            trades = []
            
            for i in range(len(df_backtest)):
                current_price = df_backtest.iloc[i]['close']
                
                # 买入逻辑
                if df_backtest.iloc[i]['buy_signal'] == 1 and position == 0:
                    position = 1
                    buy_price = current_price
                    df_backtest.iloc[i, df_backtest.columns.get_loc('position')] = 1
                    trades.append({'type': 'buy', 'price': buy_price, 'date': df_backtest.iloc[i]['date']})
                
                # 卖出逻辑
                elif df_backtest.iloc[i]['sell_signal'] == 1 and position == 1:
                    position = 0
                    sell_price = current_price
                    df_backtest.iloc[i, df_backtest.columns.get_loc('position')] = 0
                    trade_return = (sell_price - buy_price) / buy_price
                    trades.append({
                        'type': 'sell', 
                        'price': sell_price, 
                        'date': df_backtest.iloc[i]['date'],
                        'return': trade_return
                    })
                else:
                    df_backtest.iloc[i, df_backtest.columns.get_loc('position')] = position
                
                # 计算每日收益率
                if position == 1:
                    if i > 0:
                        daily_return = (current_price - df_backtest.iloc[i-1]['close']) / df_backtest.iloc[i-1]['close']
                        df_backtest.iloc[i, df_backtest.columns.get_loc('returns')] = daily_return
                
                # 计算累积收益率
                if i == 0:
                    df_backtest.iloc[i, df_backtest.columns.get_loc('cumulative_returns')] = 1.0
                else:
                    df_backtest.iloc[i, df_backtest.columns.get_loc('cumulative_returns')] = (
                        df_backtest.iloc[i-1]['cumulative_returns'] * (1 + df_backtest.iloc[i]['returns'])
                    )
            
            # 计算回测统计指标
            complete_trades = [t for t in trades if t['type'] == 'sell']
            
            if len(complete_trades) == 0:
                return {
                    'total_trades': 0,
                    'win_rate': 0,
                    'total_return': 0,
                    'max_drawdown': 0,
                    'sharpe_ratio': 0,
                    'trades': trades,
                    'df_backtest': df_backtest
                }
            
            # 总收益率
            total_return = (df_backtest['cumulative_returns'].iloc[-1] - 1) * 100
            
            # 胜率
            winning_trades = [t for t in complete_trades if t['return'] > 0]
            win_rate = len(winning_trades) / len(complete_trades) * 100 if complete_trades else 0
            
            # 最大回撤
            peak = df_backtest['cumulative_returns'].expanding().max()
            drawdown = (df_backtest['cumulative_returns'] - peak) / peak
            max_drawdown = abs(drawdown.min()) * 100
            
            # 夏普比率（简化版）
            returns_series = df_backtest['returns'][df_backtest['returns'] != 0]
            if len(returns_series) > 1:
                sharpe_ratio = returns_series.mean() / returns_series.std() * np.sqrt(252) if returns_series.std() != 0 else 0
            else:
                sharpe_ratio = 0
            
            return {
                'total_trades': len(complete_trades),
                'win_rate': win_rate,
                'total_return': total_return,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'avg_return_per_trade': np.mean([t['return'] * 100 for t in complete_trades]) if complete_trades else 0,
                'trades': trades,
                'df_backtest': df_backtest
            }
        except Exception as e:
            return None

    def plot_professional_kline(self, df, title):
        """绘制K线图 - 增加异常处理"""
        if df is None or df.empty or len(df) < 10:
            return None
            
        try:
            df = self.calc_indicators(df)
            
            # 创建信号列，但安全处理
            df['Signal'] = 0
            df['BuySignal'] = 0  # 买入信号强度
            df['SellSignal'] = 0  # 卖出信号强度
            
            # 1. MA5/MA20金叉死叉
            if 'MA5' in df.columns and 'MA20' in df.columns:
                try:
                    df.loc[(df['MA5'] > df['MA20']) & (df['MA5'].shift(1) <= df['MA20'].shift(1)), 'Signal'] = 1 
                    df.loc[(df['MA5'] < df['MA20']) & (df['MA5'].shift(1) >= df['MA20'].shift(1)), 'Signal'] = -1 
                    df.loc[df['Signal'] == 1, 'BuySignal'] = 1
                    df.loc[df['Signal'] == -1, 'SellSignal'] = 1
                except Exception:
                    pass
            
            # 2. RSI超卖反弹买入信号
            if 'RSI' in df.columns:
                try:
                    for i in range(1, len(df)):
                        if pd.notna(df.iloc[i]['RSI']) and pd.notna(df.iloc[i-1]['RSI']):
                            if df.iloc[i-1]['RSI'] < 30 and df.iloc[i]['RSI'] > 35:
                                df.iloc[i, df.columns.get_loc('BuySignal')] = max(df.iloc[i]['BuySignal'], 2)
                except Exception:
                    pass
            
            # 3. KDJ金叉买入信号
            if 'K' in df.columns and 'D' in df.columns:
                try:
                    for i in range(1, len(df)):
                        if pd.notna(df.iloc[i]['K']) and pd.notna(df.iloc[i]['D']) and \
                           pd.notna(df.iloc[i-1]['K']) and pd.notna(df.iloc[i-1]['D']):
                            if df.iloc[i-1]['K'] <= df.iloc[i-1]['D'] and df.iloc[i]['K'] > df.iloc[i]['D']:
                                if 'RSI' in df.columns and pd.notna(df.iloc[i]['RSI']) and df.iloc[i]['RSI'] > 50:
                                    df.iloc[i, df.columns.get_loc('BuySignal')] = max(df.iloc[i]['BuySignal'], 2)
                except Exception:
                    pass
            
            # 4. 布林带突破买入信号
            if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
                try:
                    for i in range(1, len(df)):
                        if pd.notna(df.iloc[i]['BB_Upper']) and pd.notna(df.iloc[i]['close']):
                            if df.iloc[i]['close'] > df.iloc[i]['BB_Upper']:
                                # 检查成交量是否放大
                                if i >= 20:
                                    vol_avg = df.iloc[i-20:i]['volume'].mean()
                                    if df.iloc[i]['volume'] > vol_avg * 1.2:
                                        df.iloc[i, df.columns.get_loc('BuySignal')] = max(df.iloc[i]['BuySignal'], 2)
                except Exception:
                    pass
            
            # 5. 200日均线趋势买入信号
            if 'MA200' in df.columns:
                try:
                    for i in range(1, len(df)):
                        if pd.notna(df.iloc[i]['MA200']) and pd.notna(df.iloc[i-1]['MA200']):
                            if df.iloc[i]['close'] > df.iloc[i]['MA200'] and df.iloc[i]['MA200'] > df.iloc[i-1]['MA200']:
                                df.iloc[i, df.columns.get_loc('BuySignal')] = max(df.iloc[i]['BuySignal'], 3)
                except Exception:
                    pass

            buy_points = df[df['BuySignal'] > 0]
            sell_points = df[df['SellSignal'] > 0]

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
            
            # 添加200日均线（如果数据足够）
            if 'MA200' in df.columns and not df['MA200'].isna().all():
                fig.add_trace(go.Scatter(x=df['date'], y=df['MA200'], name='MA200', line=dict(color='purple', width=1, dash='dash')))
            
            # 添加布林带
            if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
                try:
                    fig.add_trace(go.Scatter(x=df['date'], y=df['BB_Upper'], name='布林上轨', 
                                           line=dict(color='gray', width=1, dash='dot'), opacity=0.5))
                    fig.add_trace(go.Scatter(x=df['date'], y=df['BB_Lower'], name='布林下轨', 
                                           line=dict(color='gray', width=1, dash='dot'), opacity=0.5,
                                           fill='tonexty', fillcolor='rgba(128,128,128,0.1)'))
                except:
                    pass

            # 安全添加买卖点 - 增强版
            if not buy_points.empty:
                try:
                    # 根据信号强度分组显示
                    strong_buy = buy_points[buy_points['BuySignal'] >= 3]
                    medium_buy = buy_points[(buy_points['BuySignal'] >= 2) & (buy_points['BuySignal'] < 3)]
                    weak_buy = buy_points[buy_points['BuySignal'] == 1]
                    
                    # 强买入信号（红色，大标记）
                    if not strong_buy.empty:
                        fig.add_trace(go.Scatter(
                            x=strong_buy['date'], 
                            y=strong_buy['low']*0.97, 
                            mode='markers+text', 
                            marker=dict(symbol='triangle-up', size=16, color='red', line=dict(width=2, color='darkred')), 
                            text='强买', 
                            textposition='bottom center', 
                            name='强买入',
                            hovertemplate='<b>强买入信号</b><br>日期: %{x}<br>价格: %{y:.2f}<extra></extra>'
                        ))
                    
                    # 中等买入信号（橙色）
                    if not medium_buy.empty:
                        fig.add_trace(go.Scatter(
                            x=medium_buy['date'], 
                            y=medium_buy['low']*0.97, 
                            mode='markers+text', 
                            marker=dict(symbol='triangle-up', size=14, color='orange', line=dict(width=1, color='darkorange')), 
                            text='买入', 
                            textposition='bottom center', 
                            name='买入',
                            hovertemplate='<b>买入信号</b><br>日期: %{x}<br>价格: %{y:.2f}<extra></extra>'
                        ))
                    
                    # 弱买入信号（黄色）
                    if not weak_buy.empty:
                        fig.add_trace(go.Scatter(
                            x=weak_buy['date'], 
                            y=weak_buy['low']*0.97, 
                            mode='markers+text', 
                            marker=dict(symbol='triangle-up', size=12, color='yellow', line=dict(width=1, color='orange')), 
                            text='B', 
                            textposition='bottom center', 
                            name='金叉买入',
                            hovertemplate='<b>金叉买入</b><br>日期: %{x}<br>价格: %{y:.2f}<extra></extra>'
                        ))
                except Exception:
                    pass
            
            if not sell_points.empty:
                try:
                    fig.add_trace(go.Scatter(
                        x=sell_points['date'], 
                        y=sell_points['high']*1.03, 
                        mode='markers+text', 
                        marker=dict(symbol='triangle-down', size=12, color='green', line=dict(width=1, color='black')), 
                        text='卖出', 
                        textposition='top center', 
                        name='卖出信号',
                        hovertemplate='<b>卖出信号</b><br>日期: %{x}<br>价格: %{y:.2f}<extra></extra>'
                    ))
                except Exception:
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
if 'scan_history' not in st.session_state:
    st.session_state['scan_history'] = []

st.sidebar.header("🕹️ 控制台")

# ==========================================
# 🆕 V45 新功能区域（必须显示）
# ==========================================
st.sidebar.markdown("---")
st.sidebar.markdown("### 🆕 V45 新功能")
st.sidebar.markdown("**✨ 多策略并行扫描已启用**")

# 1. 多策略配置（直接显示，不折叠）
st.sidebar.markdown("#### 🔀 多策略扫描")
strategy_options = st.sidebar.multiselect(
    "选择策略（可多选）:",
    ["均线突破", "RSI超卖反弹", "量价背离", "KDJ金叉", "布林带突破", "温和吸筹", "换手锁仓", "妖股基因", "四星共振"],
    default=["均线突破", "RSI超卖反弹", "KDJ金叉"],
    key="strategy_selector"
)
st.session_state['selected_strategies'] = strategy_options

if not strategy_options:
    st.sidebar.warning("⚠️ 请至少选择一个策略")

comparison_mode = st.sidebar.checkbox("启用策略对比模式", value=False, key="comparison_checkbox")
st.session_state['comparison_mode'] = comparison_mode

if comparison_mode:
    st.sidebar.success("🔍 对比模式已启用")

# 2. 基本参数设置
st.sidebar.markdown("---")
st.sidebar.markdown("## ⚙️ 基本设置")

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
            # 使用缓存键
            cache_key = f"stock_list_{pool_mode}"
            if cache_key not in st.session_state:
                if pool_mode == "全市场扫描":
                    stock_list = engine.get_all_stocks()
                elif "中证500" in pool_mode:
                    index_code = "zz500"
                    stock_list = engine.get_index_stocks(index_code)
                else:
                    index_code = "hs300"
                    stock_list = engine.get_index_stocks(index_code)
                st.session_state[cache_key] = stock_list
            else:
                stock_list = st.session_state[cache_key]
            
            if stock_list:
                st.session_state['full_pool'] = stock_list 
                st.sidebar.success(f"✅ 已加载全量 {len(stock_list)} 只股票")
            else:
                st.sidebar.error("❌ 获取股票失败，请重试")
                st.sidebar.info("💡 可能的原因：\n1. 网络连接问题\n2. baostock服务暂时不可用\n3. 请稍后重试或选择其他扫描范围")
    
    if 'full_pool' in st.session_state:
        full_list = st.session_state['full_pool']
        final_code_list = full_list[:scan_limit] 
        st.sidebar.info(f"池内待扫: {len(final_code_list)} 只 (总库: {len(full_list)})")
    else:
        final_code_list = []

# 3. 历史记录功能
st.sidebar.markdown("---")
if st.session_state.get('scan_history'):
    st.sidebar.markdown("### 📚 扫描历史")
    recent_scans = st.session_state['scan_history'][:3]  # 显示最近3次
    
    for i, record in enumerate(recent_scans):
        with st.sidebar.container():
            st.sidebar.text(f"📅 {record['timestamp']}")
            st.sidebar.text(f"🎯 {record['result_count']}只 | {record['scan_type']}")
            
            col_hist1, col_hist2 = st.sidebar.columns(2)
            with col_hist1:
                if st.button("加载", key=f"load_hist_{i}"):
                    st.session_state['scan_res'] = record['results']
                    st.session_state['alerts'] = []
                    valid_options = []
                    for result in record['results']:
                        valid_options.append(f"{result['代码']} | {result['名称']}")
                    st.session_state['valid_options'] = valid_options
                    st.success(f"✅ 已加载历史记录")
                    st.rerun()
            
            with col_hist2:
                if st.button("导出", key=f"export_hist_{i}"):
                    export_df = pd.DataFrame(record['results'])
                    csv = export_df.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="CSV",
                        data=csv,
                        file_name=f"历史_{record['timestamp'].replace(':', '-').replace(' ', '_')}.csv",
                        mime="text/csv",
                        key=f"download_hist_{i}"
                    )
            st.sidebar.markdown("---")

# ==========================================
# 策略参数面板（高级设置）
# ==========================================
with st.sidebar.expander("📊 高级参数设置", expanded=False):
    st.markdown("##### 📊 技术指标参数")
    
    # 均线参数
    col1, col2 = st.columns(2)
    with col1:
        ma5_period = st.number_input("MA5周期", min_value=3, max_value=10, value=5)
        ma20_period = st.number_input("MA20周期", min_value=10, max_value=30, value=20)
    with col2:
        ma200_period = st.number_input("MA200周期", min_value=100, max_value=300, value=200)
        bb_period = st.number_input("布林带周期", min_value=10, max_value=30, value=20)
    
    st.markdown("##### 🎯 信号阈值参数")
    col3, col4 = st.columns(2)
    with col3:
        rsi_oversold = st.number_input("RSI超卖阈值", min_value=20, max_value=40, value=30)
        rsi_overbought = st.number_input("RSI超买阈值", min_value=60, max_value=80, value=70)
    with col4:
        kdj_period = st.number_input("KDJ周期", min_value=7, max_value=14, value=9)
        vol_ratio_threshold = st.number_input("成交量放大倍数", min_value=1.2, max_value=3.0, value=1.8, step=0.1)
    
    st.markdown("##### 🔥 策略参数")
    col5, col6 = st.columns(2)
    with col5:
        win_rate_threshold = st.number_input("获利筹码阈值(%)", min_value=50, max_value=90, value=70)
        limit_up_count = st.number_input("妖股基因涨停次数", min_value=2, max_value=5, value=3)
    with col6:
        turnover_threshold = st.number_input("换手率阈值(%)", min_value=3, max_value=10, value=5)
        gentle_rise_threshold = st.number_input("温和上涨阈值(%)", min_value=3, max_value=8, value=5)
    
    # 保存参数到session_state
    strategy_params = {
        'ma5_period': ma5_period,
        'ma20_period': ma20_period,
        'ma200_period': ma200_period,
        'bb_period': bb_period,
        'rsi_oversold': rsi_oversold,
        'rsi_overbought': rsi_overbought,
        'kdj_period': kdj_period,
        'vol_ratio_threshold': vol_ratio_threshold,
        'win_rate_threshold': win_rate_threshold,
        'limit_up_count': limit_up_count,
        'turnover_threshold': turnover_threshold,
        'gentle_rise_threshold': gentle_rise_threshold
    }
    st.session_state['strategy_params'] = strategy_params
    
    if st.button("🔄 重置为默认参数"):
        st.rerun()

# (历史记录部分已移动到上方直接显示区域)

# 4. 当前状态显示
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 当前状态")

# 显示当前策略配置
selected_strategies = st.session_state.get('selected_strategies', strategy_options)
st.sidebar.info(f"🎯 策略: {len(selected_strategies)}个 {'(对比模式)' if comparison_mode else ''}")

# 显示股票池状态
if 'full_pool' in st.session_state and final_code_list:
    st.sidebar.success(f"📊 准备扫描: {len(final_code_list)} 只")

# 显示历史记录数量
scan_history_count = len(st.session_state.get('scan_history', []))
if scan_history_count > 0:
    st.sidebar.text(f"📚 历史记录: {scan_history_count} 条")

st.sidebar.markdown("---")
if st.sidebar.button("🚀 启动全策略扫描 (V45)", type="primary", help="使用当前选择的策略和参数进行扫描"):
    if not final_code_list:
        st.sidebar.error("请先加载股票！")
    else:
        # 获取策略参数
        strategy_params = st.session_state.get('strategy_params', {
            'ma5_period': 5, 'ma20_period': 20, 'ma200_period': 200,
            'bb_period': 20, 'rsi_oversold': 30, 'rsi_overbought': 70,
            'kdj_period': 9, 'vol_ratio_threshold': 1.8,
            'win_rate_threshold': 70, 'limit_up_count': 3,
            'turnover_threshold': 5, 'gentle_rise_threshold': 5
        })
        
        # 获取选中的策略
        selected_strategies = st.session_state.get('selected_strategies', ["均线突破", "RSI超卖反弹", "KDJ金叉"])
        comparison_mode = st.session_state.get('comparison_mode', False)
        
        st.caption(f"当前筛选：价格 < {max_price_limit}元 | 剔除ST/科创/北交 | 模式：长连接稳定扫描")
        st.info(f"🔧 使用参数：MA{strategy_params['ma5_period']}/{strategy_params['ma20_period']}/{strategy_params['ma200_period']} | RSI{strategy_params['rsi_oversold']} | KDJ{strategy_params['kdj_period']} | 量比{strategy_params['vol_ratio_threshold']}")
        st.info(f"🎯 启用策略：{' | '.join(selected_strategies)} {'(对比模式)' if comparison_mode else ''}")
        
        # 根据是否启用多策略模式选择扫描方法
        if len(selected_strategies) > 1 or comparison_mode:
            scan_res, alerts, valid_options = engine.scan_market_multi_strategy(
                final_code_list, 
                max_price=max_price_limit, 
                selected_strategies=selected_strategies,
                strategy_params=strategy_params,
                comparison_mode=comparison_mode
            )
        else:
            scan_res, alerts, valid_options = engine.scan_market_optimized(
                final_code_list, 
                max_price=max_price_limit, 
                strategy_params=strategy_params
            )
        
        st.session_state['scan_res'] = scan_res
        st.session_state['valid_options'] = valid_options
        st.session_state['alerts'] = alerts
        st.session_state['comparison_mode_active'] = comparison_mode
        
        # 保存扫描历史记录
        scan_record = {
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'scan_type': pool_mode,
            'stock_count': len(final_code_list),
            'result_count': len(scan_res),
            'max_price': max_price_limit,
            'strategies': selected_strategies,
            'strategy_params': strategy_params.copy(),
            'results': scan_res.copy(),
            'alerts_count': len(alerts),
            'comparison_mode': comparison_mode
        }
        
        # 保持最近10条记录
        st.session_state['scan_history'].insert(0, scan_record)
        if len(st.session_state['scan_history']) > 10:
            st.session_state['scan_history'] = st.session_state['scan_history'][:10]

with st.expander("📖 **策略逻辑白皮书**", expanded=False):
    st.markdown("##### 🔍 核心策略定义")
    for k, v in STRATEGY_LOGIC.items(): st.markdown(f"- **{k}**: {v}")

st.subheader(f"⚡ 扫描结果 (价格 < {max_price_limit}元)")

# 修复：检查 scan_res 是否存在，而不是检查它是否为真值（空列表也是有效结果）
if 'scan_res' in st.session_state:
    results = st.session_state['scan_res']
    alerts = st.session_state.get('alerts', [])
    
    if alerts: 
        alert_names = "、".join(alerts[:5])  # 限制显示数量
        st.success(f"🔥 发现 {len(alerts)} 只【主力高控盘】标的：**{alert_names}**")
    
    # 修复：安全创建DataFrame，处理空结果的情况
    if results and len(results) > 0:
        try:
            df_scan = pd.DataFrame(results).sort_values(by="priority", ascending=False)
        except Exception as e:
            st.error(f"❌ 数据处理错误: {str(e)}")
            df_scan = pd.DataFrame()
    else:
        df_scan = pd.DataFrame()
    
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
        
        # 导出功能
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            # 导出CSV
            export_df = df_scan.drop('priority', axis=1, errors='ignore')
            csv = export_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📄 导出CSV",
                data=csv,
                file_name=f"选股结果_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help="导出扫描结果为CSV格式"
            )
        
        with col2:
            # 导出Excel
            try:
                import io
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    export_df.to_excel(writer, sheet_name='选股结果', index=False)
                    
                    # 获取工作表以进行格式化
                    workbook = writer.book
                    worksheet = writer.sheets['选股结果']
                    
                    # 设置列宽
                    worksheet.set_column('A:A', 10)  # 代码
                    worksheet.set_column('B:B', 12)  # 名称
                    worksheet.set_column('C:C', 15)  # 所属行业
                    worksheet.set_column('D:D', 10)  # 现价
                    worksheet.set_column('E:E', 10)  # 涨跌
                    worksheet.set_column('F:F', 12)  # 获利筹码
                    worksheet.set_column('G:G', 12)  # 风险评级
                    worksheet.set_column('H:H', 25)  # 策略信号
                    worksheet.set_column('I:I', 15)  # 综合评级
                
                excel_data = buffer.getvalue()
                st.download_button(
                    label="📊 导出Excel",
                    data=excel_data,
                    file_name=f"选股结果_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="导出扫描结果为Excel格式"
                )
            except ImportError:
                st.error("❌ 需要安装 xlsxwriter 库才能导出Excel")
        
        with col3:
            st.info(f"📊 当前结果：{len(df_scan)} 只股票 | 📅 扫描时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 策略对比分析
        if st.session_state.get('comparison_mode_active', False) and 'strategy_stats' in st.session_state:
            st.markdown("---")
            st.markdown("### 🔍 策略对比分析")
            
            strategy_stats = st.session_state['strategy_stats']
            
            # 策略效果对比
            col_compare1, col_compare2 = st.columns(2)
            
            with col_compare1:
                st.markdown("##### 📊 策略命中统计")
                strategy_comparison_data = []
                for strategy, stats in strategy_stats.items():
                    if stats['count'] > 0:
                        strategy_comparison_data.append({
                            '策略名称': strategy,
                            '命中股票数': stats['count'],
                            '命中率': f"{(stats['count'] / len(df_scan) * 100):.1f}%" if len(df_scan) > 0 else "0%"
                        })
                
                if strategy_comparison_data:
                    st.dataframe(pd.DataFrame(strategy_comparison_data), use_container_width=True)
            
            with col_compare2:
                st.markdown("##### 🏆 策略排行")
                if strategy_comparison_data:
                    sorted_strategies = sorted(strategy_comparison_data, key=lambda x: x['命中股票数'], reverse=True)
                    for i, strategy in enumerate(sorted_strategies[:5]):
                        medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i]
                        st.write(f"{medal} **{strategy['策略名称']}**: {strategy['命中股票数']}只 ({strategy['命中率']})")
            
            # 策略重叠分析
            with st.expander("🔄 策略重叠分析", expanded=False):
                st.markdown("##### 多策略命中的股票")
                multi_strategy_stocks = {}
                
                # 统计每只股票被多少个策略命中
                for result in results:
                    code = result['代码']
                    name = result['名称']
                    signals = result['策略信号'].split(' + ')
                    strategy_count = len([s for s in signals if s.strip()])
                    
                    if strategy_count > 1:
                        multi_strategy_stocks[f"{code} {name}"] = {
                            'count': strategy_count,
                            'signals': result['策略信号'],
                            'action': result['综合评级']
                        }
                
                if multi_strategy_stocks:
                    multi_df_data = []
                    for stock, info in multi_strategy_stocks.items():
                        multi_df_data.append({
                            '股票': stock,
                            '策略数量': info['count'],
                            '命中策略': info['signals'],
                            '综合评级': info['action']
                        })
                    
                    multi_df = pd.DataFrame(multi_df_data).sort_values('策略数量', ascending=False)
                    st.dataframe(multi_df, use_container_width=True)
                else:
                    st.info("暂无多策略重叠的股票")
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
                            pred_price = future_info['prices'][i]
                            price_change = pred_price - last['close']
                            price_change_pct = (price_change / last['close'] * 100) if last['close'] > 0 else 0
                            
                            # 根据涨跌设置颜色
                            delta_color = "normal"
                            if price_change_pct > 0:
                                delta_color = "inverse"
                            elif price_change_pct < 0:
                                delta_color = "normal"
                            
                            d_cols[i].metric(
                                label=future_info['dates'][i], 
                                value=f"¥{pred_price:.2f}", 
                                delta=f"{price_change:+.2f} ({price_change_pct:+.2f}%)",
                                delta_color=delta_color
                            )
                    else:
                        col2.metric("AI预测明日", f"¥{last['close']:.2f}", delta="数据不足")
                        st.warning("⚠️ 数据不足以进行AI预测，显示当前价格")
                    
                    col3.metric("数据天数", len(df))
                    
                    # K线图
                    st.markdown("### 📊 K线分析")
                    fig = engine.plot_professional_kline(df, target_name)
                    
                    if fig:
                        st.plotly_chart(fig, width='stretch')
                        st.info("""
                        💡 **图例说明**: 
                        - 🔺 **红色强买/橙色买入/黄色B** = 买入信号（红色=200日均线趋势，橙色=RSI/KDJ/布林带，黄色=MA金叉）
                        - 🔻 **绿色卖出** = 卖出信号（MA死叉）
                        - 信号仅供参考，投资需谨慎
                        """)
                    else:
                        st.warning("⚠️ 无法生成K线图，数据可能不足")
                    
                    # 回测功能
                    st.markdown("### 📈 策略回测分析")
                    
                    # 获取策略参数
                    strategy_params = st.session_state.get('strategy_params', {
                        'ma5_period': 5, 'ma20_period': 20, 'ma200_period': 200,
                        'bb_period': 20, 'rsi_oversold': 30, 'rsi_overbought': 70,
                        'kdj_period': 9, 'vol_ratio_threshold': 1.8,
                        'win_rate_threshold': 70, 'limit_up_count': 3,
                        'turnover_threshold': 5, 'gentle_rise_threshold': 5
                    })
                    
                    # 运行回测
                    backtest_result = engine.run_backtest(df, strategy_params)
                    
                    if backtest_result:
                        # 显示回测结果
                        bt_col1, bt_col2, bt_col3, bt_col4 = st.columns(4)
                        
                        with bt_col1:
                            st.metric(
                                "总收益率", 
                                f"{backtest_result['total_return']:.2f}%",
                                delta=f"vs持有: {((df.iloc[-1]['close']/df.iloc[0]['close']-1)*100 - backtest_result['total_return']):.2f}%"
                            )
                        
                        with bt_col2:
                            color = "normal" if backtest_result['win_rate'] >= 50 else "inverse"
                            st.metric("胜率", f"{backtest_result['win_rate']:.1f}%")
                        
                        with bt_col3:
                            st.metric("最大回撤", f"-{backtest_result['max_drawdown']:.2f}%")
                        
                        with bt_col4:
                            st.metric("总交易次数", backtest_result['total_trades'])
                        
                        # 回测详情
                        with st.expander("📊 详细回测数据", expanded=False):
                            bt_col5, bt_col6 = st.columns(2)
                            
                            with bt_col5:
                                st.metric("夏普比率", f"{backtest_result['sharpe_ratio']:.3f}")
                                st.metric("平均单次收益", f"{backtest_result['avg_return_per_trade']:.2f}%")
                                
                            with bt_col6:
                                buy_and_hold_return = (df.iloc[-1]['close']/df.iloc[0]['close']-1)*100
                                st.metric("买入持有收益", f"{buy_and_hold_return:.2f}%")
                                
                                alpha = backtest_result['total_return'] - buy_and_hold_return
                                st.metric("策略Alpha", f"{alpha:.2f}%")
                            
                            # 显示交易记录
                            if backtest_result['trades']:
                                st.markdown("##### 🔄 交易记录")
                                trades_data = []
                                for i, trade in enumerate(backtest_result['trades']):
                                    if trade['type'] == 'sell':
                                        # 找到对应的买入记录
                                        buy_trade = None
                                        for j in range(i-1, -1, -1):
                                            if backtest_result['trades'][j]['type'] == 'buy':
                                                buy_trade = backtest_result['trades'][j]
                                                break
                                        
                                        if buy_trade:
                                            trades_data.append({
                                                '买入日期': buy_trade['date'],
                                                '买入价格': f"¥{buy_trade['price']:.2f}",
                                                '卖出日期': trade['date'],
                                                '卖出价格': f"¥{trade['price']:.2f}",
                                                '收益率': f"{trade['return']*100:.2f}%"
                                            })
                                
                                if trades_data:
                                    st.dataframe(pd.DataFrame(trades_data), use_container_width=True)
                            
                        # 策略vs买入持有对比
                        if backtest_result['total_trades'] > 0:
                            buy_hold = (df.iloc[-1]['close'] / df.iloc[0]['close'] - 1) * 100
                            strategy_return = backtest_result['total_return']
                            
                            if strategy_return > buy_hold:
                                st.success(f"✅ 策略表现优于买入持有 {strategy_return - buy_hold:.2f}%")
                            else:
                                st.warning(f"⚠️ 策略表现低于买入持有 {abs(strategy_return - buy_hold):.2f}%")
                        else:
                            st.info("ℹ️ 在回测期间内未产生交易信号")
                    else:
                        st.error("❌ 回测计算失败，数据可能不足")
                        
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
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### 📈 数据状态")
        if 'full_pool' in st.session_state:
            st.metric("股票池总量", f"{len(st.session_state['full_pool']):,}")
        else:
            st.metric("股票池总量", "0")
    
        if 'scan_res' in st.session_state:
            st.metric("当前结果数", f"{len(st.session_state['scan_res']):,}")
        else:
            st.metric("当前结果数", "0")
    
    with col2:
        st.markdown("##### 🎯 策略状态")
        selected_strategies = st.session_state.get('selected_strategies', [])
        st.write(f"启用策略数: {len(selected_strategies)}")
        if selected_strategies:
            st.write(f"策略列表: {', '.join(selected_strategies[:2])}{'...' if len(selected_strategies) > 2 else ''}")
        
        comparison_mode = st.session_state.get('comparison_mode', False)
        st.write(f"对比模式: {'✅ 启用' if comparison_mode else '❌ 关闭'}")
    
    with col3:
        st.markdown("##### 📚 历史记录")
        scan_history = st.session_state.get('scan_history', [])
        st.metric("历史记录数", len(scan_history))
        
        if scan_history:
            latest = scan_history[0]['timestamp']
            st.write(f"最近扫描: {latest}")
    
    st.markdown("---")
    col_sys1, col_sys2 = st.columns(2)
    
    with col_sys1:
        if 'valid_options' in st.session_state:
            st.write(f"📊 可分析股票: {len(st.session_state['valid_options'])} 只")
        st.write(f"⚡ 最大扫描限制: {engine.MAX_SCAN_LIMIT:,} 只")
    
    with col_sys2:
        st.write(f"🕐 当前时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        strategy_params = st.session_state.get('strategy_params', {})
        if strategy_params:
            st.write(f"🔧 参数状态: 已自定义 ({len(strategy_params)} 项)")

# 添加使用提示
st.caption("""
💡 **V45 新功能使用指南**: 
1. **⚙️ 策略参数设置**: 调整均线周期、RSI阈值等技术指标参数，个性化您的选股策略
2. **🔀 多策略配置**: 选择多个策略并行运行，启用对比模式查看不同策略效果
3. **📈 策略回测**: 在深度分析中查看历史回测结果，包含收益率、胜率、最大回撤等关键指标
4. **📤 结果导出**: 将扫描结果导出为Excel/CSV格式，支持进一步分析
5. **📚 历史记录**: 查看和加载历史扫描记录，对比不同时期的选股结果
6. **🔍 策略对比**: 在对比模式下查看各策略命中统计和重叠分析
7. 投资有风险，所有分析结果仅供参考，请谨慎决策

""")
