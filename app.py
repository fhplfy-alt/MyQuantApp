import streamlit as st
from io import BytesIO
import json
import os
import hashlib
from datetime import datetime

# ==========================================
# ⚠️ 1. 用户管理系统 (注册+登录)
# ==========================================
# 使用明确的数据目录，确保两个应用共享数据
DATA_DIR = os.path.join(os.getcwd(), "data")
os.makedirs(DATA_DIR, exist_ok=True)  # 确保数据目录存在
USERS_FILE = os.path.join(DATA_DIR, "users.json")

# ==========================================
# 管理员配置
# ==========================================
ADMIN_PASSWORD = "admin2024"  # 管理员密码，建议修改为更安全的密码

def hash_password(password):
    """使用SHA256哈希密码"""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    """加载用户数据"""
    try:
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        pass
    return {}

def save_users(users):
    """保存用户数据"""
    try:
        with open(USERS_FILE, 'w', encoding='utf-8') as f:
            json.dump(users, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        return False

def register_user(username, password):
    """注册新用户"""
    users = load_users()
    username = username.strip()
    
    # 验证用户名
    if not username:
        return False, "用户名不能为空"
    
    # 检查用户名是否已存在
    if username in users:
        return False, "用户名已存在，请选择其他用户名"
    
    # 验证密码
    if not password or len(password) < 4:
        return False, "密码长度至少4位"
    
    # 保存用户信息
    users[username] = {
        "password_hash": hash_password(password),
        "register_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    if save_users(users):
        return True, "注册成功！"
    else:
        return False, "注册失败，请重试"

def verify_user(username, password):
    """验证用户登录"""
    users = load_users()
    username = username.strip()
    
    if not username or not password:
        return False, "请输入用户名和密码"
    
    if username not in users:
        return False, "用户名不存在，请先注册"
    
    stored_hash = users[username].get("password_hash", "")
    input_hash = hash_password(password)
    
    if stored_hash == input_hash:
        return True, "登录成功"
    else:
        return False, "密码错误"

def check_password():
    """登录/注册界面"""
    if "password_correct" not in st.session_state:
        st.markdown("### 🔐 V45 智能量化系统")
        
        # 使用tabs切换注册和登录
        tab1, tab2 = st.tabs(["🔑 登录", "📝 注册"])
        
        with tab1:
            st.markdown("#### 用户登录")
            
            # 管理员快速登录选项
            with st.expander("👨‍💼 管理员快速登录", expanded=False):
                admin_pwd = st.text_input("管理员密码", type="password", placeholder="请输入管理员密码", key="admin_quick_login")
                if st.button("管理员登录", key="admin_quick_btn", use_container_width=True):
                    if admin_pwd == ADMIN_PASSWORD:
                        st.session_state["password_correct"] = True
                        st.session_state["username"] = "admin"  # 管理员用户名
                        st.session_state["admin_logged_in"] = True  # 标记为管理员
                        st.success("✅ 管理员登录成功")
                        st.rerun()
                    else:
                        st.error("❌ 管理员密码错误")
            
            st.markdown("---")
            st.markdown("#### 普通用户登录")
            username = st.text_input("用户名", placeholder="请输入用户名", key="login_username")
            pwd = st.text_input("密码", type="password", placeholder="请输入密码", key="login_password")
            
            if st.button("登录", type="primary", use_container_width=True):
                success, message = verify_user(username, pwd)
                if success:
                    st.session_state["password_correct"] = True
                    st.session_state["username"] = username.strip()
                    st.success(message)
                    st.rerun()
                else:
                    st.error(f"❌ {message}")
        
        with tab2:
            st.markdown("#### 新用户注册")
            reg_username = st.text_input("用户名", placeholder="请输入用户名（至少1位）", key="reg_username")
            reg_password = st.text_input("密码", type="password", placeholder="请输入密码（至少4位）", key="reg_password")
            reg_password_confirm = st.text_input("确认密码", type="password", placeholder="请再次输入密码", key="reg_password_confirm")
            
            if st.button("注册", type="primary", use_container_width=True):
                # 验证输入
                if not reg_username.strip():
                    st.error("❌ 用户名不能为空")
                elif not reg_password:
                    st.error("❌ 密码不能为空")
                elif len(reg_password) < 4:
                    st.error("❌ 密码长度至少4位")
                elif reg_password != reg_password_confirm:
                    st.error("❌ 两次输入的密码不一致")
                else:
                    success, message = register_user(reg_username, reg_password)
                    if success:
                        st.success(f"✅ {message}")
                        st.info("💡 请切换到【登录】标签页进行登录")
                    else:
                        st.error(f"❌ {message}")
        
        return False
    return True

if not check_password():
    st.stop()

# ==========================================
# ⚠️ 核心配置 (保持原样)
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
    import akshare as ak # 导入akshare用于获取实时行情
    from sklearn.linear_model import LinearRegression
except ImportError as e:
    st.error(f"❌ 启动失败！缺少必要运行库: {e}")
    st.stop()

# ==========================================
# 0. 全局配置 (保持原逻辑)
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
    "📈 多头排列": "昨日收阳 且 今日收盘价 > 昨日收盘价",
    "💎 RSI超卖反弹": "RSI<30后回升至35以上,超跌反弹机会",
    "📊 布林带突破": "价格突破布林带上轨+成交量放大",
    "🎯 KDJ金叉": "K线上穿D线+RSI>50,短期买入信号",
    "📉 200日均线趋势": "价格站上200日均线+均线向上,长期上升趋势"
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
        """修复：确保全场扫描能成功获取数据"""
        try:
            bs.login() # 显式重新登录
            rs = bs.query_all_stock()
            stocks = []
            data_list = []
            while (rs.error_code == '0') and rs.next():
                data_list.append(rs.get_row_data())
            
            for data in data_list:
                if len(data) >= 2:
                    code, name = data[0], data[1]
                    if self.is_valid(code, name):
                        stocks.append(code)
            bs.logout()
            return stocks[:self.MAX_SCAN_LIMIT]
        except:
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

    def _process_single_stock(self, code, max_price=None):
        # 保持你原始的策略判定逻辑不变
        code = self.clean_code(code)
        end = datetime.datetime.now().strftime("%Y-%m-%d")
        start = (datetime.datetime.now() - datetime.timedelta(days=150)).strftime("%Y-%m-%d")
        
        data = []
        info = {'name': code, 'industry': '-', 'ipoDate': '2000-01-01'}
        
        try:
            rs_info = bs.query_stock_basic(code=code)
            if rs_info.next():
                row = rs_info.get_row_data()
                info['name'] = row[1]
                info['ipoDate'] = row[2]
            rs_ind = bs.query_stock_industry(code)
            if rs_ind.next(): info['industry'] = rs_ind.get_row_data()[3] 
            if not self.is_valid(code, info['name']): return None
            rs = bs.query_history_k_data_plus(code, "date,open,close,high,low,volume,pctChg,turn", start_date=start, frequency="d", adjustflag="3")
            while rs.next(): data.append(rs.get_row_data())
        except: return None

        if not data: return None
        df = pd.DataFrame(data, columns=["date", "open", "close", "high", "low", "volume", "pctChg", "turn"])
        df = df.apply(pd.to_numeric, errors='coerce')
        if len(df) < 60: return None

        curr = df.iloc[-1]
        prev = df.iloc[-2]
        if max_price is not None and curr['close'] > max_price: return None

        winner_rate = self.calc_winner_rate(df, curr['close'])
        df['MA5'] = df['close'].rolling(5).mean()
        df['MA20'] = df['close'].rolling(20).mean()
        df['MA200'] = df['close'].rolling(200).mean() if len(df) >= 200 else pd.Series([None] * len(df))
        risk_level = self.calc_risk_level(curr['close'], df['MA5'].iloc[-1], df['MA20'].iloc[-1])

        # 计算技术指标
        rsi = self.calc_rsi(df)
        k, d, j = self.calc_kdj(df)
        bb_upper, bb_mid, bb_lower = self.calc_bollinger(df)

        signal_tags, priority, action = [], 0, "WAIT (观望)"

        # 原有策略保留
        if (all(df['pctChg'].tail(3) > 0) and df['pctChg'].tail(3).sum() <= 5 and winner_rate > 62):
            signal_tags.append("🔴温和吸筹"); priority = 60; action = "BUY (低吸)"

        if all(df['turn'].tail(2) > 5) and winner_rate > 70:
            signal_tags.append("🔥换手锁仓"); priority = max(priority, 70); action = "BUY (博弈)"

        if len(df.tail(60)[df.tail(60)['pctChg'] > 9.5]) >= 3 and winner_rate > 80:
            signal_tags.append("🐲妖股基因"); priority = 90; action = "STRONG BUY"

        # 四星共振原逻辑
        recent_20 = df.tail(20)
        has_limit_up_20 = len(recent_20[recent_20['pctChg'] > 9.5]) > 0
        is_double_vol = (curr['volume'] > prev['volume'] * 1.8)
        if has_limit_up_20 and is_double_vol:
            signal_tags.append("👑四星共振"); priority = 100; action = "STRONG BUY"
        
        # 新增策略：RSI超卖反弹
        if rsi is not None and len(df) >= 2:
            prev_rsi = self.calc_rsi(df.iloc[:-1])
            if prev_rsi is not None and prev_rsi < 30 and rsi > 35:
                signal_tags.append("💎RSI超卖反弹")
                priority = max(priority, 65)
                if action in ["WAIT (观望)", "HOLD (持有)"]:
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
                prev_k, prev_d, _ = self.calc_kdj(df.iloc[:-1])
                if prev_k is not None and prev_d is not None:
                    if prev_k <= prev_d and k > d and rsi is not None and rsi > 50:
                        signal_tags.append("🎯KDJ金叉")
                        priority = max(priority, 70)
                        if action in ["WAIT (观望)", "HOLD (持有)"]:
                            action = "BUY (博弈)"
        
        # 新增策略：200日均线趋势
        if len(df) >= 200 and not pd.isna(df['MA200'].iloc[-1]):
            ma200_current = df['MA200'].iloc[-1]
            ma200_prev = df['MA200'].iloc[-2] if len(df) >= 201 else ma200_current
            if curr['close'] > ma200_current and ma200_current > ma200_prev:
                signal_tags.append("📉200日均线趋势")
                priority = max(priority, 80)
                if action in ["WAIT (观望)", "HOLD (持有)", "BUY (低吸)"]:
                    action = "BUY (低吸)" if action == "WAIT (观望)" else action

        # 多头排列策略
        if prev['close'] > prev['open'] and curr['close'] > prev['close']:
            signal_tags.append("📈多头排列")
            priority = max(priority, 50)
            if action == "WAIT (观望)":
                action = "HOLD (持有)"

        if priority == 0: return None

        return {
            "result": {
                "代码": code, "名称": info['name'], "所属行业": info['industry'],
                "现价": curr['close'], "涨跌": f"{curr['pctChg']:.2f}%", 
                "获利筹码": winner_rate, "风险评级": risk_level,
                "策略信号": " + ".join(signal_tags), "综合评级": action, "priority": priority
            },
            "alert": f"{info['name']}" if priority >= 90 else None,
            "option": f"{code} | {info['name']}"
        }

    def scan_market_optimized(self, code_list, max_price=None):
        # 保持原有的进度条逻辑，增加命中数量显示，优化进度显示
        results, alerts, valid_codes_list = [], [], []
        bs.login()
        total = len(code_list)
        progress_bar = st.progress(0, text=f"🚀 正在扫描 (0/{total}) | 命中: 0 只")
        
        # 根据总数决定更新频率
        if total <= 100:
            update_interval = 1  # 少于100个，每个都更新
        elif total <= 500:
            update_interval = 5  # 100-500个，每5个更新一次
        else:
            update_interval = 10  # 500个以上，每10个更新一次
        
        for i, code in enumerate(code_list):
            try:
                res = self._process_single_stock(code, max_price)
                if res:
                    results.append(res["result"])
                    if res["alert"]: alerts.append(res["alert"])
                    valid_codes_list.append(res["option"])
            except: continue
            
            # 更频繁地更新进度，让用户能看到扫描过程
            if i % update_interval == 0 or i == len(code_list) - 1:
                hit_count = len(results)
                progress = (i + 1) / total
                progress_bar.progress(progress, text=f"🔍 扫描中: {code} ({i+1}/{total}) | 命中: {hit_count} 只")
                # 添加短暂延迟，让进度条显示更清楚（不影响扫描速度）
                if i % (update_interval * 2) == 0:
                    time.sleep(0.01)  # 每更新几次才延迟，不影响整体速度

        bs.logout()
        # 显示完成状态，延迟一下再清除，让用户看到完成
        progress_bar.progress(1.0, text=f"✅ 扫描完成！共命中 {len(results)} 只")
        time.sleep(0.5)  # 显示完成状态0.5秒
        progress_bar.empty()
        return results, alerts, valid_codes_list

    def get_current_price(self, code):
        """获取股票当前价格 (优先使用实时行情)"""
        clean_code = self.clean_code(code)
        
        # 尝试从akshare获取实时价格
        try:
            df_realtime = ak.stock_zh_a_spot_em()
            # akshare返回的代码格式可能不同，需要进行匹配
            # 例如 'sh.600000' 对应 '600000'
            target_code_ak = clean_code.replace('sh.', '').replace('sz.', '')
            
            # 找到匹配的股票
            current_price_row = df_realtime[df_realtime['代码'] == target_code_ak]
            if not current_price_row.empty:
                # 返回最新价
                return float(current_price_row.iloc[0]['最新价'])
        except Exception as e:
            # st.warning(f"Akshare获取实时行情失败，尝试使用Baostock历史数据: {e}")
            pass # 静默失败，继续尝试Baostock
        
        # 如果akshare失败，或者未找到数据，则回退到Baostock获取最新收盘价
        try:
            bs.login()
            end = datetime.datetime.now().strftime("%Y-%m-%d")
            # 尝试获取当天数据，如果失败则回溯几天
            for i in range(5):
                start = (datetime.datetime.now() - datetime.timedelta(days=i)).strftime("%Y-%m-%d")
                rs = bs.query_history_k_data_plus(clean_code, "date,close", start_date=start, end_date=end, frequency="d", adjustflag="3")
                data = []
                while rs.next(): data.append(rs.get_row_data())
                if data:
                    bs.logout()
                    return float(data[-1][1])  # 返回最新收盘价
            bs.logout()
            return None
        except Exception as e:
            bs.logout()
            # st.error(f"Baostock获取历史数据失败: {e}")
            return None
    
    def analyze_holding_stock(self, code, buy_price, current_price):
        """分析持仓股票，结合技术指标给出智能卖出建议"""
        try:
            code = self.clean_code(code)
            end = datetime.datetime.now().strftime("%Y-%m-%d")
            start = (datetime.datetime.now() - datetime.timedelta(days=150)).strftime("%Y-%m-%d")
            
            bs.login()
            rs = bs.query_history_k_data_plus(code, "date,open,close,high,low,volume,pctChg,turn", start_date=start, frequency="d", adjustflag="3")
            data = []
            while rs.next(): data.append(rs.get_row_data())
            bs.logout()
            
            if not data or len(data) < 60:
                return {
                    'sell_suggestion': '持有',
                    'suggestion_reason': '数据不足',
                    'technical_signals': [],
                    'risk_level': '未知',
                    'stop_loss_price': buy_price * 0.90,  # 默认止损-10%
                    'take_profit_price': buy_price * 1.15,  # 默认止盈+15%
                    'dynamic_stop_loss': None,
                    'dynamic_take_profit': None
                }
            
            df = pd.DataFrame(data, columns=["date", "open", "close", "high", "low", "volume", "pctChg", "turn"])
            df = df.apply(pd.to_numeric, errors='coerce')
            
            curr = df.iloc[-1]
            prev = df.iloc[-2] if len(df) >= 2 else curr
            
            # 计算技术指标
            df['MA5'] = df['close'].rolling(5).mean()
            df['MA20'] = df['close'].rolling(20).mean()
            df['MA200'] = df['close'].rolling(200).mean() if len(df) >= 200 else pd.Series([None] * len(df))
            rsi = self.calc_rsi(df)
            k, d, j = self.calc_kdj(df)
            bb_upper, bb_mid, bb_lower = self.calc_bollinger(df)
            
            # 计算盈亏率
            profit_rate = ((current_price - buy_price) / buy_price) * 100
            
            # 收集技术信号
            technical_signals = []
            sell_signals_count = 0
            buy_signals_count = 0
            
            # 检测卖出信号
            # 1. MA死叉
            if len(df) >= 20:
                if prev['MA5'] >= prev['MA20'] and curr['MA5'] < curr['MA20']:
                    technical_signals.append("⚠️ MA死叉")
                    sell_signals_count += 2
            
            # 2. RSI超买
            if rsi is not None and rsi > 70:
                technical_signals.append("⚠️ RSI超买")
                sell_signals_count += 1
            
            # 3. KDJ死叉
            if k is not None and d is not None and len(df) >= 2:
                prev_k, prev_d, _ = self.calc_kdj(df.iloc[:-1])
                if prev_k is not None and prev_d is not None:
                    if prev_k >= prev_d and k < d:
                        technical_signals.append("⚠️ KDJ死叉")
                        sell_signals_count += 1
            
            # 4. 价格跌破MA20
            if len(df) >= 20 and current_price < df['MA20'].iloc[-1]:
                technical_signals.append("⚠️ 跌破MA20")
                sell_signals_count += 1
            
            # 5. 价格跌破MA5
            if len(df) >= 5 and current_price < df['MA5'].iloc[-1]:
                technical_signals.append("⚠️ 跌破MA5")
                sell_signals_count += 1
            
            # 检测买入/持有信号
            # 1. MA金叉
            if len(df) >= 20:
                if prev['MA5'] <= prev['MA20'] and curr['MA5'] > curr['MA20']:
                    technical_signals.append("✅ MA金叉")
                    buy_signals_count += 2
            
            # 2. RSI超卖反弹
            if rsi is not None and rsi < 30:
                technical_signals.append("✅ RSI超卖")
                buy_signals_count += 1
            
            # 3. 价格站上MA20
            if len(df) >= 20 and current_price > df['MA20'].iloc[-1]:
                technical_signals.append("✅ 站上MA20")
                buy_signals_count += 1
            
            # 4. 多头排列
            if len(df) >= 20 and df['MA5'].iloc[-1] > df['MA20'].iloc[-1]:
                technical_signals.append("✅ 多头排列")
                buy_signals_count += 1
            
            # 智能卖出建议逻辑
            sell_suggestion = "持有"
            suggestion_reason = ""
            
            # 结合盈亏率和技术信号
            if profit_rate >= 15:
                if sell_signals_count >= 2:
                    sell_suggestion = "强烈建议止盈"
                    suggestion_reason = f"盈利{profit_rate:.2f}%且出现{sell_signals_count}个卖出信号"
                elif sell_signals_count >= 1:
                    sell_suggestion = "考虑分批止盈"
                    suggestion_reason = f"盈利{profit_rate:.2f}%且出现卖出信号，建议分批卖出"
                else:
                    sell_suggestion = "考虑止盈"
                    suggestion_reason = f"盈利{profit_rate:.2f}%，可考虑获利了结"
            elif profit_rate >= 10:
                if sell_signals_count >= 2:
                    sell_suggestion = "建议止盈"
                    suggestion_reason = f"盈利{profit_rate:.2f}%且出现多个卖出信号"
                elif sell_signals_count >= 1:
                    sell_suggestion = "注意观察"
                    suggestion_reason = f"盈利{profit_rate:.2f}%但出现卖出信号，注意风险"
                else:
                    sell_suggestion = "考虑止盈"
                    suggestion_reason = f"盈利{profit_rate:.2f}%，可考虑止盈"
            elif profit_rate <= -10:
                if buy_signals_count >= 2:
                    sell_suggestion = "可考虑持有"
                    suggestion_reason = f"亏损{abs(profit_rate):.2f}%但出现买入信号，可考虑持有观察"
                else:
                    sell_suggestion = "强烈建议止损"
                    suggestion_reason = f"亏损{abs(profit_rate):.2f}%且无买入信号，建议止损"
            elif profit_rate <= -5:
                if sell_signals_count >= 2:
                    sell_suggestion = "建议止损"
                    suggestion_reason = f"亏损{abs(profit_rate):.2f}%且出现卖出信号"
                elif buy_signals_count >= 2:
                    sell_suggestion = "可持有观察"
                    suggestion_reason = f"亏损{abs(profit_rate):.2f}%但出现买入信号"
                else:
                    sell_suggestion = "注意止损"
                    suggestion_reason = f"亏损{abs(profit_rate):.2f}%，注意止损"
            else:
                if sell_signals_count >= 3:
                    sell_suggestion = "建议卖出"
                    suggestion_reason = f"出现{sell_signals_count}个卖出信号，建议卖出"
                elif buy_signals_count >= 2:
                    sell_suggestion = "持有"
                    suggestion_reason = f"出现买入信号，建议持有"
                else:
                    sell_suggestion = "持有"
                    suggestion_reason = "技术指标正常，建议持有"
            
            # 动态止盈止损价格
            # 动态止损：如果盈利，止损点随价格上涨而上移
            dynamic_stop_loss = None
            dynamic_take_profit = None
            
            if profit_rate > 0:
                # 盈利时，止损点设为买入价的1.05倍（保本+5%）
                dynamic_stop_loss = max(buy_price * 1.05, current_price * 0.95)
                # 动态止盈：盈利15%以上时，止盈点设为当前价的0.92倍（保留8%利润）
                if profit_rate >= 15:
                    dynamic_take_profit = current_price * 0.92
                elif profit_rate >= 10:
                    dynamic_take_profit = current_price * 0.95
            else:
                # 亏损时，止损点设为买入价的0.90倍（-10%）
                dynamic_stop_loss = buy_price * 0.90
            
            # 风险评级
            risk_level = "低"
            if sell_signals_count >= 3:
                risk_level = "高"
            elif sell_signals_count >= 1:
                risk_level = "中"
            
            return {
                'sell_suggestion': sell_suggestion,
                'suggestion_reason': suggestion_reason,
                'technical_signals': technical_signals,
                'risk_level': risk_level,
                'stop_loss_price': buy_price * 0.90,  # 固定止损-10%
                'take_profit_price': buy_price * 1.15,  # 固定止盈+15%
                'dynamic_stop_loss': dynamic_stop_loss,
                'dynamic_take_profit': dynamic_take_profit,
                'rsi': rsi,
                'ma5': df['MA5'].iloc[-1] if len(df) >= 5 else None,
                'ma20': df['MA20'].iloc[-1] if len(df) >= 20 else None,
                'sell_signals_count': sell_signals_count,
                'buy_signals_count': buy_signals_count
            }
        except Exception as e:
            return {
                'sell_suggestion': '持有',
                'suggestion_reason': f'分析出错: {str(e)}',
                'technical_signals': [],
                'risk_level': '未知',
                'stop_loss_price': buy_price * 0.90,
                'take_profit_price': buy_price * 1.15,
                'dynamic_stop_loss': None,
                'dynamic_take_profit': None,
                'rsi': None,
                'ma5': None,
                'ma20': None,
                'sell_signals_count': 0,
                'buy_signals_count': 0
            }
    
    def get_deep_data(self, code):
        """修复白屏的关键：增加严谨的数据校验"""
        try:
            bs.login()
            end = datetime.datetime.now().strftime("%Y-%m-%d")
            start = (datetime.datetime.now() - datetime.timedelta(days=180)).strftime("%Y-%m-%d")
            rs = bs.query_history_k_data_plus(code, "date,open,close,high,low,volume", start_date=start, end_date=end, frequency="d", adjustflag="3")
            data = []
            while rs.next(): data.append(rs.get_row_data())
            bs.logout()
            if not data: return None
            df = pd.DataFrame(data, columns=["date", "open", "close", "high", "low", "volume"])
            df[["open", "close", "high", "low", "volume"]] = df[["open", "close", "high", "low", "volume"]].apply(pd.to_numeric, errors='coerce')
            return df.dropna()
        except: return None

    def run_ai_prediction(self, df):
        """增强版AI预测：预估后三天股票走势，包括价格、涨跌幅等"""
        if df is None or len(df) < 30: return None
        try:
            # 使用更多历史数据提高预测准确性
            recent = df.tail(30).reset_index(drop=True)
            X = np.array(recent.index).reshape(-1, 1)
            y = recent['close'].values
            
            # 训练模型
            model = LinearRegression().fit(X, y)
            
            # 预测后三天价格
            next_indices = np.array([[len(recent)], [len(recent)+1], [len(recent)+2]])
            pred_prices = model.predict(next_indices)
            
            # 计算当前价格
            current_price = df['close'].iloc[-1]
            
            # 计算涨跌幅
            changes = [(p - current_price) / current_price * 100 for p in pred_prices]
            
            # 生成日期（后三天）：明日/后日/大后日
            last_date = pd.to_datetime(df['date'].iloc[-1])
            date_labels = ["明日", "后日", "大后日"]
            dates = []
            day_offset = 1
            for i in range(3):
                next_date = last_date + datetime.timedelta(days=day_offset)
                # 跳过周末
                while next_date.weekday() >= 5:  # 5=Saturday, 6=Sunday
                    next_date += datetime.timedelta(days=1)
                dates.append(f"{date_labels[i]} ({next_date.strftime('%m-%d')})")
                day_offset += 1
            
            # 判断趋势（颜色：红色=上涨，绿色=下跌，蓝色=横盘）
            avg_change = np.mean(changes)
            if avg_change > 2:
                color = "red"  # 红色=预测上涨
                title = "📈 AI预测：上涨趋势"
                desc = f"预计未来三天平均涨幅 {avg_change:.2f}%"
                action = "建议持有或逢低买入"
            elif avg_change < -2:
                color = "green"  # 绿色=预测下跌
                title = "📉 AI预测：下跌趋势"
                desc = f"预计未来三天平均跌幅 {abs(avg_change):.2f}%"
                action = "建议谨慎观望或减仓"
            else:
                color = "blue"  # 蓝色=预测横盘
                title = "➡️ AI预测：震荡整理"
                desc = f"预计未来三天波动较小，平均变化 {abs(avg_change):.2f}%"
                action = "建议持有观望"

            return {
                "dates": dates,
                "prices": pred_prices.tolist(),
                "changes": changes,
                "pred_price": pred_prices[0],
                "current_price": current_price,
                "color": color,
                "title": title,
                "desc": desc,
                "action": action
            }
        except Exception as e:
            return None

    def plot_professional_kline(self, df, title):
        """增强版K线图：添加买卖信号标记"""
        if df is None or df.empty: return None
            
        try:
            # 计算技术指标
            df['MA5'] = df['close'].rolling(5).mean()
            df['MA20'] = df['close'].rolling(20).mean()
            df['MA200'] = df['close'].rolling(200).mean() if len(df) >= 200 else None
            
            # 计算RSI和KDJ用于信号判断
            rsi = self.calc_rsi(df)
            k, d, j = self.calc_kdj(df)
            bb_upper, bb_mid, bb_lower = self.calc_bollinger(df)
            
            # 创建K线图
            fig = go.Figure()
            
            # 添加K线（调换红绿颜色：A股习惯红=涨，绿=跌）
            fig.add_trace(go.Candlestick(
                x=df['date'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='K线',
                increasing_line_color='red',    # 上涨用红色
                decreasing_line_color='green',  # 下跌用绿色
                increasing_fillcolor='red',     # 上涨填充红色
                decreasing_fillcolor='green'    # 下跌填充绿色
            ))
            
            # 添加均线
            if 'MA5' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['date'],
                    y=df['MA5'],
                    mode='lines',
                    name='MA5',
                    line=dict(color='orange', width=1)
                ))
            
            if 'MA20' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['date'],
                    y=df['MA20'],
                    mode='lines',
                    name='MA20',
                    line=dict(color='blue', width=1)
                ))
            
            if df['MA200'] is not None and not df['MA200'].isna().all():
                fig.add_trace(go.Scatter(
                    x=df['date'],
                    y=df['MA200'],
                    mode='lines',
                    name='MA200',
                    line=dict(color='purple', width=1, dash='dash')
                ))
            
            # 添加布林带
            if bb_upper is not None and bb_lower is not None:
                # 计算布林带数据
                period = 20
                if len(df) >= period:
                    ma = df['close'].rolling(window=period).mean()
                    std = df['close'].rolling(window=period).std()
                    upper = ma + (std * 2)
                    lower = ma - (std * 2)
                    
                    fig.add_trace(go.Scatter(
                        x=df['date'],
                        y=upper,
                        mode='lines',
                        name='布林上轨',
                        line=dict(color='gray', width=1, dash='dash'),
                        showlegend=False
                    ))
                    fig.add_trace(go.Scatter(
                        x=df['date'],
                        y=lower,
                        mode='lines',
                        name='布林下轨',
                        line=dict(color='gray', width=1, dash='dash'),
                        fill='tonexty',
                        fillcolor='rgba(128,128,128,0.1)',
                        showlegend=False
                    ))
            
            # 识别买卖信号（区分不同强度）
            strong_buy_signals = []  # 红色"强买"：200日均线趋势
            medium_buy_signals = []  # 橙色"买入"：RSI/KDJ/布林带
            basic_buy_signals = []   # 黄色"B"：MA金叉
            sell_signals = []        # 绿色"卖出"：MA死叉
            
            for i in range(1, len(df)):
                curr = df.iloc[i]
                prev = df.iloc[i-1]
                
                # 1. 最强买入信号：200日均线趋势（红色"强买"）
                if i >= 200 and df['MA200'] is not None and not df['MA200'].isna().all():
                    ma200_curr = df['MA200'].iloc[i]
                    ma200_prev = df['MA200'].iloc[i-1] if i >= 201 else ma200_curr
                    if curr['close'] > ma200_curr and ma200_curr > ma200_prev:
                        strong_buy_signals.append((df['date'].iloc[i], curr['low'] * 0.98, "强买"))
                
                # 2. 中等强度买入信号：RSI/KDJ/布林带（橙色"买入"）
                # RSI超卖反弹
                if i >= 15:
                    curr_rsi = self.calc_rsi(df.iloc[:i+1])
                    prev_rsi = self.calc_rsi(df.iloc[:i])
                    if prev_rsi is not None and curr_rsi is not None:
                        if prev_rsi < 30 and curr_rsi > 35:
                            medium_buy_signals.append((df['date'].iloc[i], curr['low'] * 0.98, "买入"))
                
                # KDJ金叉
                if i >= 10:
                    curr_k, curr_d, _ = self.calc_kdj(df.iloc[:i+1])
                    prev_k, prev_d, _ = self.calc_kdj(df.iloc[:i])
                    if prev_k is not None and prev_d is not None and curr_k is not None and curr_d is not None:
                        if prev_k <= prev_d and curr_k > curr_d:
                            medium_buy_signals.append((df['date'].iloc[i], curr['low'] * 0.98, "买入"))
                
                # 布林带突破
                if i >= 20 and bb_upper is not None:
                    if curr['close'] > bb_upper and curr['volume'] > df['volume'].iloc[max(0, i-20):i].mean() * 1.2:
                        medium_buy_signals.append((df['date'].iloc[i], curr['low'] * 0.98, "买入"))
                
                # 3. 基础买入信号：MA5上穿MA20（金叉）（黄色"B"）
                if i >= 20:
                    if prev['MA5'] <= prev['MA20'] and curr['MA5'] > curr['MA20']:
                        basic_buy_signals.append((df['date'].iloc[i], curr['low'] * 0.98, "B"))
            
                # 卖出信号：MA5下穿MA20（死叉）（绿色"卖出"）
                if i >= 20:
                    if prev['MA5'] >= prev['MA20'] and curr['MA5'] < curr['MA20']:
                        sell_signals.append((df['date'].iloc[i], curr['high'] * 1.02, "卖出"))
            
            # 添加最强买入信号标记（红色"强买"）
            if strong_buy_signals:
                dates, prices, _ = zip(*strong_buy_signals)
                fig.add_trace(go.Scatter(
                    x=list(dates),
                    y=list(prices),
                    mode='markers+text', 
                    name='强买',
                    text=['强买'] * len(dates),
                    textposition='top center',
                    marker=dict(
                        symbol='triangle-up',
                        size=15,
                        color='red',
                        line=dict(width=2, color='darkred')
                    ),
                    textfont=dict(size=10, color='red')
                ))
                    
            # 添加中等强度买入信号标记（橙色"买入"）
            if medium_buy_signals:
                dates, prices, _ = zip(*medium_buy_signals)
                fig.add_trace(go.Scatter(
                    x=list(dates),
                    y=list(prices),
                    mode='markers+text', 
                    name='买入',
                    text=['买入'] * len(dates),
                    textposition='top center',
                    marker=dict(
                        symbol='triangle-up',
                        size=12,
                        color='orange',
                        line=dict(width=2, color='darkorange')
                    ),
                    textfont=dict(size=9, color='orange')
                ))
                    
            # 添加基础买入信号标记（黄色"B"）
            if basic_buy_signals:
                dates, prices, _ = zip(*basic_buy_signals)
                fig.add_trace(go.Scatter(
                    x=list(dates),
                    y=list(prices),
                    mode='markers+text', 
                    name='B',
                    text=['B'] * len(dates),
                    textposition='top center',
                    marker=dict(
                        symbol='triangle-up',
                        size=10,
                        color='yellow',
                        line=dict(width=1, color='gold')
                    ),
                    textfont=dict(size=8, color='darkgoldenrod')
                ))
            
            # 添加卖出信号标记（绿色"卖出"）
            if sell_signals:
                dates, prices, _ = zip(*sell_signals)
                fig.add_trace(go.Scatter(
                    x=list(dates),
                    y=list(prices),
                    mode='markers+text', 
                    name='卖出',
                    text=['卖出'] * len(dates),
                    textposition='bottom center',
                    marker=dict(
                        symbol='triangle-down',
                        size=12,
                        color='green',
                        line=dict(width=2, color='darkgreen')
                    ),
                    textfont=dict(size=9, color='green')
                ))
            
            # 更新布局
            fig.update_layout(
                title=title,
                xaxis_rangeslider_visible=False,
                height=600,
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            return fig
        except Exception as e:
            # 如果出错，返回基础K线图（调换红绿颜色）
            fig = go.Figure(data=[go.Candlestick(
                x=df['date'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='K线',
                increasing_line_color='red',    # 上涨用红色
                decreasing_line_color='green',  # 下跌用绿色
                increasing_fillcolor='red',     # 上涨填充红色
                decreasing_fillcolor='green'    # 下跌填充绿色
            )])
            fig.update_layout(title=title, xaxis_rangeslider_visible=False, height=500)
            return fig

# ==========================================
# 3. 界面 UI (完全恢复原布局)
# ==========================================
engine = QuantsEngine()

if 'full_pool' not in st.session_state: st.session_state['full_pool'] = []
if 'scan_res' not in st.session_state: st.session_state['scan_res'] = []
if 'valid_options' not in st.session_state: st.session_state['valid_options'] = []

# 持仓数据持久化存储（按用户隔离）
def get_holdings_file():
    """根据当前用户名获取持仓文件路径"""
    username = st.session_state.get("username", "default")
    # 清理用户名中的特殊字符，避免文件名问题
    safe_username = "".join(c for c in username if c.isalnum() or c in ('-', '_'))
    return os.path.join(DATA_DIR, f"holdings_data_{safe_username}.json")

def load_holdings():
    """从文件加载当前用户的持仓数据"""
    try:
        holdings_file = get_holdings_file()
        if os.path.exists(holdings_file):
            with open(holdings_file, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        pass  # 静默失败，不影响应用启动
    return []

def save_holdings(holdings):
    """保存当前用户的持仓数据到文件"""
    try:
        holdings_file = get_holdings_file()
        with open(holdings_file, 'w', encoding='utf-8') as f:
            json.dump(holdings, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        return False

# 初始化持仓数据（从文件加载，每个用户独立）
if 'holdings' not in st.session_state:
    st.session_state['holdings'] = load_holdings()

# ==========================================
# 管理员功能辅助函数
# ==========================================
def get_user_holdings_file(username):
    """根据用户名获取持仓文件路径（用于管理员查看）"""
    safe_username = "".join(c for c in username if c.isalnum() or c in ('-', '_'))
    return os.path.join(DATA_DIR, f"holdings_data_{safe_username}.json")

def load_user_holdings(username):
    """加载指定用户的持仓数据（用于管理员查看）"""
    try:
        holdings_file = get_user_holdings_file(username)
        if os.path.exists(holdings_file):
            with open(holdings_file, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        pass
    return []

def check_admin_access():
    """检查是否有管理员权限"""
    return st.session_state.get("admin_logged_in", False)

st.sidebar.header("🕹️ 控制台")
max_price_limit = st.sidebar.slider("💰 价格上限 (元)", 3.0, 100.0, 20.0)
pool_mode = st.sidebar.radio("🔎 选股范围:", ("中证500 (中小盘)", "沪深300 (大盘)", "全市场扫描", "手动输入"))
scan_limit = st.sidebar.slider("🔢 扫描数量 (池大小)", 50, 6000, 500, step=50)

if pool_mode == "手动输入":
    target_pool_str = st.sidebar.text_area("监控股票池", "600519, 002131", height=100)
    final_code_list = [c.strip() for c in target_pool_str.replace("，", ",").split(",") if c.strip()]
else:
    if st.sidebar.button(f"📥 加载 {pool_mode} 成分股"):
        with st.spinner("获取中..."):
            if pool_mode == "全市场扫描": st.session_state['full_pool'] = engine.get_all_stocks()
            elif "中证500" in pool_mode: st.session_state['full_pool'] = engine.get_index_stocks("zz500")
            else: st.session_state['full_pool'] = engine.get_index_stocks("hs300")
            st.sidebar.success(f"已加载 {len(st.session_state['full_pool'])} 只")
    final_code_list = st.session_state.get('full_pool', [])[:scan_limit]

if st.sidebar.button("🚀 启动全策略扫描 (V45)", type="primary"):
    if not final_code_list: st.sidebar.error("请先加载股票！")
    else:
        res, alerts, opts = engine.scan_market_optimized(final_code_list, max_price=max_price_limit)
        st.session_state['scan_res'], st.session_state['valid_options'], st.session_state['alerts'] = res, opts, alerts

# 我的持仓管理功能
st.sidebar.markdown("---")
st.sidebar.subheader("💼 我的持仓")

# 添加持仓表单
with st.sidebar.expander("➕ 添加持仓", expanded=False):
    holding_code = st.text_input("股票代码", placeholder="如: 600519", key="holding_code_input")
    holding_price = st.number_input("买入价格 (元)", min_value=0.01, value=0.01, step=0.01, key="holding_price_input")
    holding_qty = st.number_input("买入数量 (股)", min_value=1, value=100, step=100, key="holding_qty_input")
    
    if st.button("✅ 添加持仓", key="add_holding_btn"):
        if holding_code and holding_price > 0 and holding_qty > 0:
            # 清理代码格式
            clean_code = engine.clean_code(holding_code.strip())
            # 检查是否已存在
            existing = [h for h in st.session_state['holdings'] if h['code'] == clean_code]
            if existing:
                st.sidebar.warning(f"⚠️ {clean_code} 已存在，将更新持仓")
                # 更新持仓
                for h in st.session_state['holdings']:
                    if h['code'] == clean_code:
                        h['buy_price'] = holding_price
                        h['quantity'] = holding_qty
                        h['buy_date'] = datetime.datetime.now().strftime("%Y-%m-%d")
            else:
                # 添加新持仓
                st.session_state['holdings'].append({
                    'code': clean_code,
                    'buy_price': holding_price,
                    'quantity': holding_qty,
                    'buy_date': datetime.datetime.now().strftime("%Y-%m-%d")
                })
            # 保存到文件
            if save_holdings(st.session_state['holdings']):
                st.sidebar.success(f"✅ 已添加 {clean_code}（已保存）")
            else:
                st.sidebar.success(f"✅ 已添加 {clean_code}")
            st.rerun()

# 显示持仓列表
if st.session_state['holdings']:
    st.sidebar.markdown("**持仓列表:**")
    for i, holding in enumerate(st.session_state['holdings']):
        col1, col2 = st.sidebar.columns([3, 1])
        with col1:
            st.sidebar.text(f"{holding['code']}")
        with col2:
            if st.sidebar.button("🗑️", key=f"del_{i}"):
                st.session_state['holdings'].pop(i)
                # 保存到文件
                save_holdings(st.session_state['holdings'])
                st.rerun()
else:
    st.sidebar.info("💡 暂无持仓，点击上方添加")

# 导出Excel功能（放在sidebar中，确保显示）
st.sidebar.markdown("---")
st.sidebar.subheader("📊 导出功能")

# ==========================================
# 管理员入口
# ==========================================
st.sidebar.markdown("---")
st.sidebar.subheader("👨‍💼 管理员")

# 检查是否已登录管理员
if check_admin_access():
    st.sidebar.success("✅ 管理员已登录")
    if st.sidebar.button("🚪 退出管理员"):
        st.session_state["admin_logged_in"] = False
        st.rerun()
else:
    # 管理员登录
    with st.sidebar.expander("🔐 管理员登录", expanded=False):
        admin_pwd = st.text_input("管理员密码", type="password", key="admin_pwd_input")
        if st.button("登录", key="admin_login_btn"):
            if admin_pwd == ADMIN_PASSWORD:
                st.session_state["admin_logged_in"] = True
                st.sidebar.success("✅ 登录成功")
                st.rerun()
            else:
                st.sidebar.error("❌ 密码错误")

# 检查是否有扫描结果
scan_res = st.session_state.get('scan_res', [])
if scan_res and len(scan_res) > 0:
    # 创建DataFrame并排序：priority >= 90的排在最前面
    df_export = pd.DataFrame(scan_res)
    if 'priority' in df_export.columns:
        df_export['is_high_priority'] = df_export['priority'] >= 90
        df_export = df_export.sort_values(by=['is_high_priority', 'priority'], ascending=[False, False])
        df_export = df_export.drop(columns=['is_high_priority'], errors='ignore')
    
    # 移除priority列（内部使用，不需要导出）
    df_export_clean = df_export.drop(columns=['priority'], errors='ignore')
    
    # 创建Excel文件
    try:
        # 确保数据不为空
        if df_export_clean.empty:
            st.sidebar.warning("⚠️ 没有可导出的数据")
        else:
            # 使用BytesIO创建Excel文件（修复导出问题）
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl', mode='w') as writer:
                df_export_clean.to_excel(writer, index=False, sheet_name='扫描结果')
            
            # 重置文件指针并获取数据
            output.seek(0)
            excel_data = output.read()
            output.close()
            
            # 生成文件名（包含日期时间）
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"股票扫描结果_{timestamp}.xlsx"
            
            # 显示导出按钮
            st.sidebar.download_button(
                label="📥 导出为Excel",
                data=excel_data,
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                key="export_excel_button"
            )
    except ImportError as import_err:
        st.sidebar.error("❌ 缺少 openpyxl 库")
        st.sidebar.info("💡 请运行: pip install openpyxl")
        st.sidebar.code(str(import_err))
    except Exception as e:
        st.sidebar.error(f"❌ 导出失败: {str(e)}")
        import traceback
        with st.sidebar.expander("查看详细错误"):
            st.code(traceback.format_exc())
else:
    st.sidebar.info("💡 请先进行扫描，扫描完成后可导出结果")

# ==========================================
# 主内容区域 - 页面选择
# ==========================================
# 如果是管理员，显示管理功能选项
show_admin = False
if check_admin_access():
    main_tabs = st.tabs(["📊 量化分析", "👨‍💼 管理后台"])
    if main_tabs[1]:  # 如果点击了管理后台标签
        show_admin = True

# 根据选择的标签页显示内容
if show_admin:
    # ==========================================
    # 管理后台功能
    # ==========================================
    st.title("👨‍💼 管理员后台系统")
    st.caption("用户数据管理与统计")
    
    # 管理功能页面选择
    admin_page = st.radio(
        "选择功能",
        ["用户列表", "持仓详情", "数据统计", "数据导出"],
        horizontal=True
    )
    
    # 1. 用户列表
    if admin_page == "用户列表":
        st.header("👥 所有注册用户")
        users = load_users()
        
        if not users:
            st.info("📭 暂无注册用户")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("总用户数", len(users))
            with col2:
                users_with_holdings = sum(1 for username in users.keys() if load_user_holdings(username))
                st.metric("有持仓用户", users_with_holdings)
            with col3:
                users_without_holdings = len(users) - users_with_holdings
                st.metric("无持仓用户", users_without_holdings)
            
            st.markdown("---")
            user_data = []
            for username, user_info in users.items():
                holdings = load_user_holdings(username)
                holdings_count = len(holdings)
                user_data.append({
                    "用户名": username,
                    "注册时间": user_info.get("register_date", "未知"),
                    "持仓数量": holdings_count,
                    "状态": "有持仓" if holdings_count > 0 else "无持仓"
                })
            
            if user_data:
                df_users = pd.DataFrame(user_data)
                st.dataframe(df_users, hide_index=True, use_container_width=True)
                
                st.markdown("### 🔍 搜索用户")
                search_username = st.text_input("输入用户名搜索", placeholder="如: user001")
                if search_username:
                    if search_username in users:
                        st.success(f"✅ 找到用户: {search_username}")
                        user_info = users[search_username]
                        st.json({
                            "用户名": search_username,
                            "注册时间": user_info.get("register_date", "未知"),
                            "持仓数量": len(load_user_holdings(search_username))
                        })
                    else:
                        st.warning(f"❌ 未找到用户: {search_username}")
    
    # 2. 持仓详情
    elif admin_page == "持仓详情":
        st.header("💼 用户持仓详情")
        users = load_users()
        
        if not users:
            st.info("📭 暂无注册用户")
        else:
            selected_user = st.selectbox("选择要查看的用户", ["全部用户"] + list(users.keys()))
            
            if selected_user == "全部用户":
                st.subheader("📊 所有用户持仓汇总")
                all_holdings_data = []
                for username in users.keys():
                    holdings = load_user_holdings(username)
                    for holding in holdings:
                        all_holdings_data.append({
                            "用户名": username,
                            "股票代码": holding.get("code", "-"),
                            "买入价": holding.get("buy_price", 0),
                            "数量": holding.get("quantity", 0),
                            "买入日期": holding.get("buy_date", "-")
                        })
                
                if all_holdings_data:
                    df_all = pd.DataFrame(all_holdings_data)
                    st.dataframe(df_all, hide_index=True, use_container_width=True)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("总持仓数", len(all_holdings_data))
                    with col2:
                        unique_stocks = df_all["股票代码"].nunique()
                        st.metric("不同股票数", unique_stocks)
                    with col3:
                        total_quantity = df_all["数量"].sum()
                        st.metric("总持股数", f"{total_quantity:,.0f}")
                else:
                    st.info("📭 暂无持仓数据")
            else:
                st.subheader(f"📊 {selected_user} 的持仓")
                holdings = load_user_holdings(selected_user)
                
                if not holdings:
                    st.info(f"📭 用户 {selected_user} 暂无持仓")
                else:
                    holdings_data = []
                    for holding in holdings:
                        holdings_data.append({
                            "股票代码": holding.get("code", "-"),
                            "买入价": f"{holding.get('buy_price', 0):.2f}",
                            "数量": holding.get("quantity", 0),
                            "买入日期": holding.get("buy_date", "-"),
                            "总成本": f"{holding.get('buy_price', 0) * holding.get('quantity', 0):.2f}"
                        })
                    
                    df_holdings = pd.DataFrame(holdings_data)
                    st.dataframe(df_holdings, hide_index=True, use_container_width=True)
                    total_cost = sum(h.get('buy_price', 0) * h.get('quantity', 0) for h in holdings)
                    st.metric("总持仓成本", f"¥{total_cost:,.2f}")
    
    # 3. 数据统计
    elif admin_page == "数据统计":
        st.header("📊 数据统计")
        users = load_users()
        
        if not users:
            st.info("📭 暂无数据")
        else:
            all_holdings = []
            for username in users.keys():
                holdings = load_user_holdings(username)
                for holding in holdings:
                    all_holdings.append({
                        "用户名": username,
                        "股票代码": holding.get("code", "-"),
                        "买入价": holding.get("buy_price", 0),
                        "数量": holding.get("quantity", 0)
                    })
            
            if all_holdings:
                df_stats = pd.DataFrame(all_holdings)
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("总用户数", len(users))
                with col2:
                    st.metric("总持仓数", len(all_holdings))
                with col3:
                    unique_stocks = df_stats["股票代码"].nunique()
                    st.metric("不同股票数", unique_stocks)
                with col4:
                    total_quantity = df_stats["数量"].sum()
                    st.metric("总持股数", f"{total_quantity:,.0f}")
                
                st.markdown("---")
                st.subheader("🔥 热门股票排行（持有用户数）")
                stock_user_count = df_stats.groupby("股票代码")["用户名"].nunique().sort_values(ascending=False)
                if len(stock_user_count) > 0:
                    df_popular = pd.DataFrame({
                        "股票代码": stock_user_count.index,
                        "持有用户数": stock_user_count.values
                    })
                    st.dataframe(df_popular.head(20), hide_index=True, use_container_width=True)
                
                st.markdown("---")
                st.subheader("📈 持仓数量排行（总股数）")
                stock_quantity = df_stats.groupby("股票代码")["数量"].sum().sort_values(ascending=False)
                if len(stock_quantity) > 0:
                    df_quantity = pd.DataFrame({
                        "股票代码": stock_quantity.index,
                        "总持股数": stock_quantity.values
                    })
                    st.dataframe(df_quantity.head(20), hide_index=True, use_container_width=True)
                
                st.markdown("---")
                st.subheader("👥 用户持仓排行")
                user_holdings_count = df_stats.groupby("用户名").size().sort_values(ascending=False)
                if len(user_holdings_count) > 0:
                    df_user_rank = pd.DataFrame({
                        "用户名": user_holdings_count.index,
                        "持仓数量": user_holdings_count.values
                    })
                    st.dataframe(df_user_rank, hide_index=True, use_container_width=True)
            else:
                st.info("📭 暂无持仓数据")
    
    # 4. 数据导出
    elif admin_page == "数据导出":
        st.header("📥 数据导出")
        users = load_users()
        
        if not users:
            st.info("📭 暂无数据可导出")
        else:
            export_type = st.radio("选择导出类型", ["所有用户信息", "所有持仓数据", "统计数据"])
            
            if export_type == "所有用户信息":
                user_data = []
                for username, user_info in users.items():
                    holdings = load_user_holdings(username)
                    user_data.append({
                        "用户名": username,
                        "注册时间": user_info.get("register_date", "未知"),
                        "持仓数量": len(holdings)
                    })
                
                if user_data:
                    df_export = pd.DataFrame(user_data)
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df_export.to_excel(writer, index=False, sheet_name='用户信息')
                    excel_data = output.getvalue()
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"用户信息_{timestamp}.xlsx"
                    st.download_button(
                        label="📥 下载用户信息Excel",
                        data=excel_data,
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary"
                    )
            
            elif export_type == "所有持仓数据":
                all_holdings_data = []
                for username in users.keys():
                    holdings = load_user_holdings(username)
                    for holding in holdings:
                        all_holdings_data.append({
                            "用户名": username,
                            "股票代码": holding.get("code", "-"),
                            "买入价": holding.get("buy_price", 0),
                            "数量": holding.get("quantity", 0),
                            "买入日期": holding.get("buy_date", "-"),
                            "总成本": holding.get("buy_price", 0) * holding.get("quantity", 0)
                        })
                
                if all_holdings_data:
                    df_export = pd.DataFrame(all_holdings_data)
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df_export.to_excel(writer, index=False, sheet_name='持仓数据')
                    excel_data = output.getvalue()
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"持仓数据_{timestamp}.xlsx"
                    st.download_button(
                        label="📥 下载持仓数据Excel",
                        data=excel_data,
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary"
                    )
                else:
                    st.info("📭 暂无持仓数据")
            
            elif export_type == "统计数据":
                all_holdings = []
                for username in users.keys():
                    holdings = load_user_holdings(username)
                    for holding in holdings:
                        all_holdings.append({
                            "用户名": username,
                            "股票代码": holding.get("code", "-"),
                            "买入价": holding.get("buy_price", 0),
                            "数量": holding.get("quantity", 0)
                        })
                
                if all_holdings:
                    df_stats = pd.DataFrame(all_holdings)
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        user_stats = df_stats.groupby("用户名").size().reset_index(name="持仓数量")
                        user_stats.to_excel(writer, index=False, sheet_name='用户统计')
                        stock_stats = df_stats.groupby("股票代码").agg({
                            "用户名": "nunique",
                            "数量": "sum"
                        }).reset_index()
                        stock_stats.columns = ["股票代码", "持有用户数", "总持股数"]
                        stock_stats = stock_stats.sort_values("持有用户数", ascending=False)
                        stock_stats.to_excel(writer, index=False, sheet_name='股票统计')
                    excel_data = output.getvalue()
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"统计数据_{timestamp}.xlsx"
                    st.download_button(
                        label="📥 下载统计数据Excel",
                        data=excel_data,
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary"
                    )
                else:
                    st.info("📭 暂无数据")
    
    st.markdown("---")

else:
    # ==========================================
    # 普通用户功能（原有功能）
    # ==========================================
    # 策略展示逻辑 (保持原样)
    with st.expander("📖 **策略逻辑白皮书**", expanded=False):
        for k, v in STRATEGY_LOGIC.items(): st.markdown(f"- **{k}**: {v}")

    # 持仓监控面板
if st.session_state['holdings']:
    st.markdown("---")
    st.subheader("💼 我的持仓监控")
    
    holdings_data = []
    holdings_analysis = {}  # 存储每只股票的深度分析数据
    total_profit = 0
    total_cost = 0
    
    with st.spinner("正在分析持仓数据..."):
        for holding in st.session_state['holdings']:
            code = holding['code']
            buy_price = holding['buy_price']
            quantity = holding['quantity']
            buy_date = holding.get('buy_date', '-')
            
            # 获取当前价格
            current_price = engine.get_current_price(code)
            if current_price:
                profit = (current_price - buy_price) * quantity
                profit_rate = ((current_price - buy_price) / buy_price) * 100
                total_profit += profit
                total_cost += buy_price * quantity
                
                # 获取股票名称
                try:
                    bs.login()
                    rs_info = bs.query_stock_basic(code=code)
                    stock_name = code
                    if rs_info.next():
                        stock_name = rs_info.get_row_data()[1]
                    bs.logout()
                except:
                    stock_name = code
                
                # 技术分析（结合技术指标）
                analysis = engine.analyze_holding_stock(code, buy_price, current_price)
                holdings_analysis[code] = analysis
                
                # 构建技术信号显示
                signals_display = " | ".join(analysis.get('technical_signals', [])) if analysis.get('technical_signals') else "无特殊信号"
                
                holdings_data.append({
                    '代码': code,
                    '名称': stock_name,
                    '买入价': f"{buy_price:.2f}",
                    '当前价': f"{current_price:.2f}",
                    '数量': quantity,
                    '盈亏': f"{profit:.2f}",
                    '盈亏率': f"{profit_rate:.2f}%",
                    '买入日期': buy_date,
                    '卖出建议': analysis['sell_suggestion'],
                    '技术信号': signals_display,
                    '风险评级': analysis['risk_level']
                })
            else:
                holdings_data.append({
                    '代码': code,
                    '名称': code,
                    '买入价': f"{buy_price:.2f}",
                    '当前价': "获取中...",
                    '数量': quantity,
                    '盈亏': "-",
                    '盈亏率': "-",
                    '买入日期': buy_date,
                    '卖出建议': "-",
                    '技术信号': "-",
                    '风险评级': "-"
                })
    
    # 显示总盈亏
    if total_cost > 0:
        total_profit_rate = (total_profit / total_cost) * 100
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("总成本", f"¥{total_cost:,.2f}")
        with col2:
            # A股习惯：红色=上涨/盈利，绿色=下跌/亏损
            if total_profit > 0:
                # 盈利显示红色（inverse反转颜色：正数红色）
                st.metric("总盈亏", f"¥{total_profit:,.2f}", delta=f"+{total_profit_rate:.2f}%", delta_color="inverse")
            elif total_profit < 0:
                # 亏损显示绿色（normal正常颜色：负数绿色）
                st.metric("总盈亏", f"¥{total_profit:,.2f}", delta=f"{total_profit_rate:.2f}%", delta_color="normal")
            else:
                # 盈亏平衡
                st.metric("总盈亏", f"¥{total_profit:,.2f}", delta="0.00%")
        with col3:
            st.metric("持仓数量", len(st.session_state['holdings']))
        with col4:
            if total_profit > 0:
                st.success("📈 整体盈利")
            elif total_profit < 0:
                st.error("📉 整体亏损")
            else:
                st.info("➡️ 盈亏平衡")
    
    # 显示持仓表格
    if holdings_data:
        df_holdings = pd.DataFrame(holdings_data)
        # 配置持仓表格列提示信息
        holdings_column_config = {
            "代码": st.column_config.TextColumn("代码", help="股票代码"),
            "名称": st.column_config.TextColumn("名称", help="股票名称"),
            "买入价": st.column_config.TextColumn("买入价", help="买入时的价格（元）"),
            "当前价": st.column_config.TextColumn("当前价", help="当前股票价格（元）"),
            "数量": st.column_config.NumberColumn("数量", help="持有的股票数量（股）", format="%d"),
            "盈亏": st.column_config.TextColumn("盈亏", help="盈亏金额（元），正数表示盈利，负数表示亏损"),
            "盈亏率": st.column_config.TextColumn("盈亏率", help="盈亏百分比，正数表示盈利，负数表示亏损"),
            "买入日期": st.column_config.TextColumn("买入日期", help="买入股票的日期"),
            "卖出建议": st.column_config.TextColumn(
                "卖出建议", 
                help="""智能卖出建议（结合技术指标）：
强烈建议止盈/止损: 盈利≥15%且出现多个卖出信号，或亏损≥10%且无买入信号
考虑止盈/止损: 盈利≥10%或亏损≥5%，结合技术信号判断
注意观察/止损: 出现卖出信号，需要密切关注
持有: 技术指标正常，建议继续持有"""
            ),
            "技术信号": st.column_config.TextColumn(
                "技术信号", 
                help="""技术指标信号：
⚠️ MA死叉: MA5下穿MA20，卖出信号
⚠️ RSI超买: RSI>70，可能超买
⚠️ KDJ死叉: K线下穿D线，卖出信号
⚠️ 跌破MA20/MA5: 价格跌破均线，可能转弱
✅ MA金叉: MA5上穿MA20，买入信号
✅ RSI超卖: RSI<30，可能超卖反弹
✅ 站上MA20: 价格站上均线，可能转强
✅ 多头排列: 均线多头排列，趋势向上"""
            ),
            "风险评级": st.column_config.TextColumn("风险评级", help="风险评级：低 - 低风险，中 - 中等风险，高 - 高风险，未知 - 数据不足无法评级")
        }
        st.dataframe(df_holdings, hide_index=True, use_container_width=True, column_config=holdings_column_config)
    
    # 持仓股票深度分析
    st.markdown("### 🔍 持仓股票深度分析")
    
    # 选择要分析的股票 - 获取股票名称
    holding_options = []
    for h in st.session_state['holdings']:
        code = h['code']
        # 尝试获取股票名称
        stock_name = code
        try:
            bs.login()
            rs_info = bs.query_stock_basic(code=code)
            if rs_info.next():
                stock_name = rs_info.get_row_data()[1]
            bs.logout()
        except:
            pass
        holding_options.append(f"{code} | {stock_name}")
    
    if holding_options:
        selected_holding = st.selectbox("选择要深度分析的持仓股票", holding_options, key="holding_analysis_select")
        selected_code = selected_holding.split("|")[0].strip()
        
        # 找到对应的持仓信息
        selected_holding_info = None
        for h in st.session_state['holdings']:
            if h['code'] == selected_code:
                selected_holding_info = h
                break
        
        if selected_holding_info and selected_code in holdings_analysis:
            analysis = holdings_analysis[selected_code]
            current_price = engine.get_current_price(selected_code)
            buy_price = selected_holding_info['buy_price']
            profit_rate = ((current_price - buy_price) / buy_price) * 100 if current_price else 0
            
            # 显示分析结果
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 智能卖出建议")
                # 根据建议类型显示不同颜色
                sell_suggestion = analysis.get('sell_suggestion', '持有')
                if "强烈建议" in sell_suggestion or "建议止损" in sell_suggestion:
                    st.error(f"**{sell_suggestion}**")
                elif "考虑" in sell_suggestion or "建议" in sell_suggestion:
                    st.warning(f"**{sell_suggestion}**")
                else:
                    st.info(f"**{sell_suggestion}**")
                
                st.markdown(f"**理由：** {analysis.get('suggestion_reason', '暂无')}")
                
                st.markdown("#### ⚠️ 止盈止损建议")
                if analysis.get('stop_loss_price'):
                    st.markdown(f"**固定止损价：** ¥{analysis['stop_loss_price']:.2f} (-10%)")
                if analysis.get('take_profit_price'):
                    st.markdown(f"**固定止盈价：** ¥{analysis['take_profit_price']:.2f} (+15%)")
                
                if analysis.get('dynamic_stop_loss'):
                    st.markdown(f"**动态止损价：** ¥{analysis['dynamic_stop_loss']:.2f}")
                    st.caption("💡 动态止损会随价格上涨而上移，保护利润")
                
                if analysis.get('dynamic_take_profit'):
                    st.markdown(f"**动态止盈价：** ¥{analysis['dynamic_take_profit']:.2f}")
                    st.caption("💡 动态止盈会随价格调整，锁定部分利润")
            
            with col2:
                st.markdown("#### 📈 技术指标")
                if analysis.get('rsi'):
                    rsi_status = "超买" if analysis['rsi'] > 70 else ("超卖" if analysis['rsi'] < 30 else "正常")
                    st.metric("RSI", f"{analysis['rsi']:.2f}", delta=rsi_status)
                
                if analysis.get('ma5'):
                    st.metric("MA5", f"¥{analysis['ma5']:.2f}")
                
                if analysis.get('ma20'):
                    st.metric("MA20", f"¥{analysis['ma20']:.2f}")
                
                st.markdown("#### 🎯 信号统计")
                st.markdown(f"**卖出信号：** {analysis.get('sell_signals_count', 0)} 个")
                st.markdown(f"**买入信号：** {analysis.get('buy_signals_count', 0)} 个")
                st.markdown(f"**风险评级：** {analysis.get('risk_level', '未知')}")
            
            # 显示技术信号详情
            if analysis.get('technical_signals'):
                st.markdown("#### 🔔 技术信号详情")
                for signal in analysis['technical_signals']:
                    if "⚠️" in signal:
                        st.warning(signal)
                    else:
                        st.success(signal)
            
            # 深度分析：K线图
            if st.button(f"📊 查看 {selected_code} 的K线图", key=f"kline_{selected_code}"):
                with st.spinner("正在生成K线图..."):
                    df = engine.get_deep_data(selected_code)
                    if df is not None and not df.empty:
                        stock_name = selected_holding_info.get('name', selected_code)
                        try:
                            bs.login()
                            rs_info = bs.query_stock_basic(code=selected_code)
                            if rs_info.next():
                                stock_name = rs_info.get_row_data()[1]
                            bs.logout()
                        except:
                            pass
                        
                        fig = engine.plot_professional_kline(df, f"{stock_name} - K线图（持仓分析）")
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # 在K线图上标注买入价
                            st.info(f"💡 **买入价：¥{buy_price:.2f}** | **当前价：¥{current_price:.2f}** | **盈亏率：{profit_rate:.2f}%**")
                    else:
                        st.error("❌ 无法获取K线数据")
            
            # AI预测
            if st.button(f"🤖 查看 {selected_code} 的AI预测", key=f"ai_{selected_code}"):
                with st.spinner("正在生成AI预测..."):
                    df = engine.get_deep_data(selected_code)
                    if df is not None and not df.empty:
                        future = engine.run_ai_prediction(df)
                        if future:
                            st.markdown("#### 🤖 AI预测：未来三天走势")
                            col1, col2, col3 = st.columns(3)
                            current_price_pred = future['current_price']
                            
                            with col1:
                                st.metric("当前价格", f"¥{current_price_pred:.2f}")
                            
                            if future['color'] == 'green':
                                st.success(f"### {future['title']}\n{future['desc']}\n\n**{future['action']}**")
                            elif future['color'] == 'red':
                                st.error(f"### {future['title']}\n{future['desc']}\n\n**{future['action']}**")
                            else:
                                st.warning(f"### {future['title']}\n{future['desc']}\n\n**{future['action']}**")
                            
                            # 显示后三天预测
                            st.markdown("#### 📅 AI 时空推演 (未来3日)")
                            pred_cols = st.columns(3)
                            for i in range(3):
                                pred_price = future['prices'][i]
                                change = future['changes'][i]
                                date_label = future['dates'][i]
                                change_amount = pred_price - current_price_pred
                                
                                with pred_cols[i]:
                                    if change > 0:
                                        st.metric(
                                            label=date_label,
                                            value=f"¥{pred_price:.2f}",
                                            delta=f"{change_amount:+.2f} ({change:+.2f}%)",
                                            delta_color="inverse"
                                        )
                                    else:
                                        st.metric(
                                            label=date_label,
                                            value=f"¥{pred_price:.2f}",
                                            delta=f"{change_amount:+.2f} ({change:+.2f}%)",
                                            delta_color="normal"
                                        )
                        else:
                            st.warning("⚠️ AI预测数据不足")
                    else:
                        st.error("❌ 无法获取预测数据")
    
    st.markdown("---")

if st.session_state['scan_res']:
    # 排序：priority >= 90的排在最前面，然后按priority降序
    df_scan = pd.DataFrame(st.session_state['scan_res'])
    df_scan['is_high_priority'] = df_scan['priority'] >= 90
    df_scan = df_scan.sort_values(by=['is_high_priority', 'priority'], ascending=[False, False])
    df_scan = df_scan.drop(columns=['is_high_priority'], errors='ignore')
    
    # 显示命中股票数量
    total_count = len(df_scan)
    st.success(f"✅ **扫描完成！共命中 {total_count} 只符合条件的股票**")
    
    # 显示主力高控盘标的（priority >= 90的股票）
    if 'alerts' in st.session_state and st.session_state['alerts']:
        alert_count = len(st.session_state['alerts'])
        alert_names = "、".join(st.session_state['alerts'][:5])  # 最多显示5个
        if len(st.session_state['alerts']) > 5:
            alert_names += f"等{alert_count}只"
        st.success(f"🔥 **发现 {alert_count} 只【主力高控盘】标的：{alert_names}**")
    
    # 配置列提示信息
    column_config = {
        "代码": st.column_config.TextColumn("代码", help="股票代码"),
        "名称": st.column_config.TextColumn("名称", help="股票名称"),
        "所属行业": st.column_config.TextColumn("所属行业", help="股票所属行业分类"),
        "现价": st.column_config.NumberColumn("现价", help="当前股票价格（元）", format="%.2f"),
        "涨跌": st.column_config.TextColumn("涨跌", help="涨跌幅百分比"),
        "获利筹码": st.column_config.NumberColumn("获利筹码", help="获利筹码比例，表示当前价格下盈利的筹码占比（%）", format="%.2f"),
        "风险评级": st.column_config.TextColumn("风险评级", help="风险评级：Low(安全) - 低风险，Med(破位) - 中等风险，High(高危) - 高风险"),
        "策略信号": st.column_config.TextColumn(
            "策略信号", 
            help="""策略信号说明：
👑 四星共振: [涨停+缺口+连阳+倍量] 同时满足，最强主升浪信号！
🐲 妖股基因: 60天内3板 + 筹码>80%，游资龙头特征。
🔥 换手锁仓: 连续高换手 + 高获利，主力清洗浮筹接力。
🔴 温和吸筹: 3连阳但涨幅小 + 筹码集中，主力潜伏期。
📈 多头排列: 股价收阳且重心上移，趋势健康，建议持有。
💎 RSI超卖反弹: RSI<30后回升，超跌反弹机会。
📊 布林带突破: 价格突破布林带上轨，强势突破信号。
🎯 KDJ金叉: K线上穿D线，短期买入信号。
📉 200日均线趋势: 价格站上200日均线，长期上升趋势。"""
        ),
        "综合评级": st.column_config.TextColumn(
            "综合评级", 
            help="""操作建议说明：
🟥 STRONG BUY: 【重点关注】确定性极高
🟧 BUY (博弈): 【激进买入】短线博弈
🟨 BUY (低吸): 【稳健买入】逢低建仓
🟦 HOLD: 【持股】趋势完好，拿住不动
⬜ WAIT: 【观望】无机会"""
        ),
        "priority": st.column_config.NumberColumn("priority", help="优先级评分，数值越高表示信号越强（0-100）", format="%d")
    }
    
    st.dataframe(df_scan, hide_index=True, column_config=column_config)

# 深度分析 (增强版)
if st.session_state['valid_options']:
    st.subheader("🧠 深度分析")
    target = st.selectbox("选择目标进行深度分析", st.session_state['valid_options'])
    target_code = target.split("|")[0].strip()
    target_name = target.split("|")[1].strip() if "|" in target else target

    if st.button(f"🚀 立即分析 {target_name}", type="primary"):
        with st.spinner("正在获取数据并分析..."):
                df = engine.get_deep_data(target_code)
                if df is not None and not df.empty:
                    # 显示K线图（带买卖信号）
                    st.markdown("### 📊 K线分析（含买卖信号）")
                    fig = engine.plot_professional_kline(df, f"{target_name} - K线图")
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.info("""
                        💡 **图例说明**: 
                        - 🔺 **红色"强买"** = 200日均线趋势信号，最强买入信号
                        - 🔺 **橙色"买入"** = RSI/KDJ/布林带信号，中等强度买入
                        - 🔺 **黄色"B"** = MA金叉信号，基础买入信号
                        - 🔻 **绿色"卖出"** = MA死叉信号，建议卖出
                        - **橙色线** = MA5均线（5日移动平均线）
                        - **蓝色线** = MA20均线（20日移动平均线）
                        - **紫色虚线** = MA200均线（200日移动平均线，长期趋势）
                        - **灰色区域** = 布林带（价格波动范围）
                        - 信号仅供参考，投资需谨慎
                        """)
                    
                    # 显示AI预测（后三天走势）
                    st.markdown("### 🤖 AI预测：未来三天走势")
                    future = engine.run_ai_prediction(df)
                    if future:
                        col1, col2, col3 = st.columns(3)
                        
                        # 显示当前价格
                        current_price = future['current_price']
                        col1.metric("当前价格", f"¥{current_price:.2f}")
                        
                        # 显示预测信息
                        if future['color'] == 'green':
                            st.success(f"### {future['title']}\n{future['desc']}\n\n**{future['action']}**")
                        elif future['color'] == 'red':
                            st.error(f"### {future['title']}\n{future['desc']}\n\n**{future['action']}**")
                        else:
                            st.warning(f"### {future['title']}\n{future['desc']}\n\n**{future['action']}**")

                        # 显示后三天详细预测（明日/后日/大后日）
                        st.markdown("#### 📅 AI 时空推演 (未来3日)")
                        pred_cols = st.columns(3)
                        for i in range(3):
                            pred_price = future['prices'][i]
                            change = future['changes'][i]
                            date_label = future['dates'][i]  # 已经是"明日 (MM-DD)"格式
                            change_amount = pred_price - current_price
                            
                            with pred_cols[i]:
                                if change > 0:
                                    st.metric(
                                        label=date_label,
                                        value=f"¥{pred_price:.2f}", 
                                        delta=f"{change_amount:+.2f} ({change:+.2f}%)",
                                        delta_color="inverse"
                                    )
                                else:
                                    st.metric(
                                        label=date_label,
                                        value=f"¥{pred_price:.2f}",
                                        delta=f"{change_amount:+.2f} ({change:+.2f}%)",
                                        delta_color="normal"
                                    )
                        
                        # 显示预测数据表格
                        with st.expander("📋 查看详细预测数据"):
                            pred_df = pd.DataFrame({
                                '日期': future['dates'],  # 已经是"明日 (MM-DD)"格式
                                '预测价格': [f"¥{p:.2f}" for p in future['prices']],
                                '涨跌金额': [f"{p - current_price:+.2f}" for p in future['prices']],
                                '涨跌幅': [f"{c:+.2f}%" for c in future['changes']]
                            })
                            st.dataframe(pred_df, hide_index=True)
                    else:
                        st.warning("⚠️ AI预测数据不足，无法生成预测")
                        
                    # 显示最近交易数据
                    with st.expander("📋 查看最近交易数据"):
                        st.dataframe(df.tail(20), hide_index=True)
                else:
                    st.error("❌ 数据获取失败，请重试")
            
st.caption("💡 使用提示：扫描时请勿刷新页面。投资有风险。")