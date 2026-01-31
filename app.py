import streamlit as st
from io import BytesIO
import json
import os
import hashlib
import datetime

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
        "register_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
    from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
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
        # 缓存列名检测结果，避免每次调用都检测
        self._realtime_code_column = None
        self._realtime_price_column = None
        self._realtime_columns_checked = False
        # 数据源优先级配置（针对短期交易，优先使用更实时的数据源）
        self.price_data_sources = [
            'akshare_spot_em',      # akshare东方财富实时行情（最常用）
            'akshare_spot',          # akshare实时行情（备选）
            'akshare_spot_sina',     # akshare新浪实时行情（备选）
        ]
        # 基本信息缓存：避免对命中股票重复查询（保持原功能不变，仅减少重复IO）
        # key: code(str), value: (name, industry, ipoDate)
        self._basic_info_cache = {}
    
    def safe_bs_login(self, max_retries=3):
        """安全登录baostock，带重试机制"""
        for attempt in range(max_retries):
            try:
                result = bs.login()
                if result.error_code == '0':
                    return True
            except Exception:
                pass
            if attempt < max_retries - 1:
                time.sleep(0.5)  # 重试前等待0.5秒
        return False
    
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
            if not self.safe_bs_login():
                return []
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
    
    def calc_macd(self, df, fast=12, slow=26, signal=9):
        """计算MACD指标（DIF, DEA, MACD柱）"""
        try:
            if len(df) < slow + signal:
                return None, None, None
            # 计算EMA
            ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
            ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
            # DIF = EMA12 - EMA26
            dif = ema_fast - ema_slow
            # DEA = DIF的9日EMA
            dea = dif.ewm(span=signal, adjust=False).mean()
            # MACD柱 = (DIF - DEA) * 2
            macd_histogram = (dif - dea) * 2
            return dif.iloc[-1], dea.iloc[-1], macd_histogram.iloc[-1]
        except:
            return None, None, None
    
    def check_macd_signal(self, df):
        """检查MACD强势信号：DIF和DEA均在零轴上方，且当日DIF上穿DEA（金叉）"""
        try:
            if len(df) < 35:  # 至少需要26+9=35天数据
                return False
            dif, dea, _ = self.calc_macd(df)
            if dif is None or dea is None:
                return False
            # 计算前一天的DIF和DEA用于判断金叉
            if len(df) < 36:
                return False
            prev_dif, prev_dea, _ = self.calc_macd(df.iloc[:-1])
            if prev_dif is None or prev_dea is None:
                return False
            # 条件：DIF和DEA均在零轴上方，且当日DIF上穿DEA（金叉）
            if dif > 0 and dea > 0 and prev_dif <= prev_dea and dif > dea:
                return True
            return False
        except:
            return False
    
    def check_volume_anomaly(self, df):
        """检查成交量异动信号：当日成交量>过去5日平均成交量的2倍，且收盘价>过去5日最高价（放量突破）"""
        try:
            if len(df) < 6:  # 至少需要6天数据（5天用于计算平均值+1天当日）
                return False
            curr = df.iloc[-1]
            # 过去5日平均成交量
            avg_volume_5d = df['volume'].tail(5).iloc[:-1].mean()  # 排除当日，取前5日
            if avg_volume_5d <= 0:
                return False
            # 过去5日最高价（排除当日）
            max_high_5d = df['high'].tail(5).iloc[:-1].max()
            # 条件：当日成交量>过去5日平均成交量的2倍，且收盘价>过去5日最高价
            if curr['volume'] > avg_volume_5d * 2 and curr['close'] > max_high_5d:
                return True
            return False
        except:
            return False
    
    def is_high_position_risk(self, df):
        """判断股票是否处于高位且缩量，可能存在主力出货风险"""
        if df is None or len(df) < 60:
            return False
        
        try:
            close = df['close'].iloc[-1]
            high_60 = df['high'].rolling(window=60, min_periods=1).max().iloc[-1]
            
            # 条件1：股价接近60日高点（>90%）
            near_high = close > high_60 * 0.9
            
            # 条件2：当前成交量 < 5日均量的60%（明显缩量）
            vol = df['volume'].iloc[-1]
            vol_ma5 = df['volume'].rolling(window=5, min_periods=1).mean().iloc[-1]
            low_volume = vol < vol_ma5 * 0.6
            
            return near_high and low_volume
        except:
            return False  # 异常时默认不过滤（安全优先）
    
    def get_hot_concept_stocks(self, top_n=10):
        """获取当日热门概念板块中的股票
        
        Args:
            top_n: 获取前N个热门概念板块，默认10个
            
        Returns:
            list: 股票代码列表，如果失败则返回空列表
        """
        try:
            # 获取概念板块列表
            concept_df = ak.stock_board_concept_em()
            if concept_df is None or concept_df.empty:
                return []
            
            # 按涨跌幅排序，取前top_n个热门概念
            if '涨跌幅' in concept_df.columns:
                concept_df = concept_df.sort_values('涨跌幅', ascending=False)
            elif '涨跌' in concept_df.columns:
                concept_df = concept_df.sort_values('涨跌', ascending=False)
            
            top_concepts = concept_df.head(top_n)
            
            # 收集所有概念板块中的股票代码
            all_stocks = set()
            for idx, row in top_concepts.iterrows():
                try:
                    concept_name = row.get('板块名称', '') or row.get('名称', '')
                    if not concept_name:
                        continue
                    
                    # 获取该概念板块的成分股
                    cons_df = ak.stock_board_concept_cons_em(symbol=concept_name)
                    if cons_df is not None and not cons_df.empty:
                        # 提取股票代码列
                        code_col = None
                        for col in ['代码', '股票代码', 'code', 'symbol']:
                            if col in cons_df.columns:
                                code_col = col
                                break
                        
                        if code_col:
                            for code in cons_df[code_col]:
                                if pd.notna(code) and code:
                                    # 标准化代码格式
                                    clean_code = self.clean_code(str(code).strip())
                                    all_stocks.add(clean_code)
                except Exception:
                    continue
            
            return list(all_stocks)
        except Exception:
            return []  # 网络请求失败时返回空列表，自动回退到全市场扫描
    
    def get_hot_concepts(self, top_n=8):
        """获取当日热门概念板块名称列表
        
        Args:
            top_n: 获取前N个热门概念板块，默认8个
            
        Returns:
            list: 概念板块名称列表，如果失败则返回空列表
        """
        try:
            # 获取概念板块列表
            concept_df = ak.stock_board_concept_em()
            if concept_df is None or concept_df.empty:
                return []
            
            # 按涨跌幅排序，取前top_n个热门概念
            if '涨跌幅' in concept_df.columns:
                concept_df = concept_df.sort_values('涨跌幅', ascending=False)
            elif '涨跌' in concept_df.columns:
                concept_df = concept_df.sort_values('涨跌', ascending=False)
            
            top_concepts = concept_df.head(top_n)
            
            # 提取概念板块名称
            concept_names = []
            for idx, row in top_concepts.iterrows():
                concept_name = row.get('板块名称', '') or row.get('名称', '')
                if concept_name:
                    concept_names.append(concept_name)
            
            return concept_names
        except Exception:
            return []  # 网络请求失败时返回空列表
    
    def get_stocks_in_concept(self, concept_name):
        """获取指定概念板块中的股票代码列表
        
        Args:
            concept_name: 概念板块名称
            
        Returns:
            list: 股票代码列表（原始格式，未clean），如果失败则返回空列表
        """
        try:
            # 获取该概念板块的成分股
            cons_df = ak.stock_board_concept_cons_em(symbol=concept_name)
            if cons_df is None or cons_df.empty:
                return []
            
            # 提取股票代码列
            code_col = None
            for col in ['代码', '股票代码', 'code', 'symbol']:
                if col in cons_df.columns:
                    code_col = col
                    break
            
            if code_col:
                stocks = []
                for code in cons_df[code_col]:
                    if pd.notna(code) and code:
                        stocks.append(str(code).strip())
                return stocks
            
            return []
        except Exception:
            return []  # 获取失败时返回空列表
    
    def get_main_force_net_inflow(self, code):
        """获取股票的主力资金净流入（单位：元）
        
        Args:
            code: 股票代码（已clean格式，如 'sh.600000' 或 'sz.000001'）
            
        Returns:
            float: 主力资金净流入（元），如果获取失败返回0
        """
        try:
            # code 是 clean_code 后的格式，如 'sh.600000' 或 'sz.000001'
            # 需要提取6位数字代码
            code_str = str(code).replace('sh.', '').replace('sz.', '').strip()
            
            # 转换为 akshare 需要的格式：'600000' -> 'sh600000'
            if code_str.startswith(('60', '68')):
                ak_symbol = f"sh{code_str}"
            else:
                ak_symbol = f"sz{code_str}"
            
            df = ak.stock_individual_fund_flow(symbol=ak_symbol)
            if df is not None and not df.empty:
                net_inflow = pd.to_numeric(df['主力净流入'].iloc[0], errors='coerce')
                return net_inflow if pd.notna(net_inflow) else 0
        except Exception as e:
            print(f"获取主力资金流失败 ({code}): {e}")
        return 0

    def _process_single_stock(self, code, max_price=None, realtime_data_cache=None, price_map=None):
        """处理单只股票的策略分析
        
        性能优化说明：
        1. 支持价格映射表，避免重复匹配
        2. 提前使用实时价格过滤，减少不必要的baostock查询
        3. 保持原有策略判定逻辑不变
        
        Args:
            code: 股票代码
            max_price: 最大价格限制
            realtime_data_cache: 实时行情数据缓存
            price_map: 代码到价格的映射表（可选，用于优化性能）
        """
        # 注意：该函数会访问baostock（网络IO），在批量扫描场景下性能瓶颈主要在这里。
        # scan_market_optimized 已改为：主线程串行拉取历史数据 + 线程池并行做指标计算，
        # 从而在不破坏baostock稳定性的前提下提升速度。
        code = self.clean_code(code)
        
        # 如果有价格映射表且设置了价格上限，先检查实时价格
        if max_price is not None and price_map is not None and code in price_map:
            cached_price = price_map[code]
            if cached_price is not None and cached_price > max_price:
                return None  # 提前过滤，避免后续查询
        
        end = datetime.datetime.now().strftime("%Y-%m-%d")
        start = (datetime.datetime.now() - datetime.timedelta(days=150)).strftime("%Y-%m-%d")
        
        data = []
        info = {'name': code, 'industry': '-', 'ipoDate': '2000-01-01'}
        
        try:
            rs = bs.query_history_k_data_plus(
                code,
                "date,open,close,high,low,volume,pctChg,turn",
                start_date=start,
                frequency="d",
                adjustflag="3"
            )
            while rs.next():
                data.append(rs.get_row_data())
        except:
            return None

        analysis = self._analyze_single_stock_from_history(
            code=code,
            data=data,
            max_price=max_price,
            realtime_data_cache=realtime_data_cache,
            price_map=price_map
        )
        if not analysis:
            return None

        # 仅对命中股票查询展示用信息（并做缓存），避免无效IO
        name, industry, ipo_date = self._get_basic_info_cached(code)
        if not self.is_valid(code, name):
            return None

        return {
            "result": {
                "代码": code,
                "名称": name,
                "所属行业": industry,
                "现价": analysis["display_price"],
                "涨跌": analysis["pct_chg"],
                "获利筹码": analysis["winner_rate"],
                "风险评级": analysis["risk_level"],
                "策略信号": analysis["signals"],
                "综合评级": analysis["action"],
                "priority": analysis["priority"]
            },
            "alert": f"{name}" if analysis["priority"] >= 90 else None,
            "option": f"{code} | {name}"
        }

    def _analyze_single_stock_from_history(self, code, data, max_price=None, realtime_data_cache=None, price_map=None, allow_realtime_price=True):
        """从历史K线数据中计算策略信号（纯计算逻辑，便于并发）

        说明：
        - 该方法不访问baostock，只做DataFrame构建与指标计算
        - scan_market_optimized 会“主线程串行拉取历史数据 + 线程池并行计算”，以兼顾稳定性与速度
        """
        if not data or len(data) < 60:
            return None

        try:
            last_close = float(data[-1][2])
            if max_price is not None and last_close > max_price:
                return None
        except (ValueError, IndexError):
            pass

        df = pd.DataFrame(
            data,
            columns=["date", "open", "close", "high", "low", "volume", "pctChg", "turn"]
        )
        df = df.apply(pd.to_numeric, errors='coerce')
        if len(df) < 60:
            return None

        curr = df.iloc[-1]
        prev = df.iloc[-2]

        # 过滤器1：检查是否处于高位缩量风险（在计算信号前过滤）
        try:
            if self.is_high_position_risk(df):
                return None  # 跳过该股票
        except:
            pass  # 异常时默认不过滤（安全优先）

        winner_rate = self.calc_winner_rate(df, curr['close'])
        df['MA5'] = df['close'].rolling(5).mean()
        df['MA20'] = df['close'].rolling(20).mean()
        df['MA200'] = df['close'].rolling(200).mean() if len(df) >= 200 else pd.Series([None] * len(df))
        risk_level = self.calc_risk_level(curr['close'], df['MA5'].iloc[-1], df['MA20'].iloc[-1])

        rsi = self.calc_rsi(df)
        k, d, _j = self.calc_kdj(df)
        bb_upper, _bb_mid, bb_lower = self.calc_bollinger(df)

        signal_tags, priority, action = [], 0, "WAIT (观望)"

        # 计算放量确认条件（用于增强激进信号可信度）
        try:
            vol_today = df['volume'].iloc[-1]
            vol_ma5 = df['volume'].rolling(5).mean().iloc[-2] if len(df) >= 6 else 0
            has_volume_confirmation = vol_today > vol_ma5 * 1.5 if vol_ma5 > 0 else False
        except:
            has_volume_confirmation = False  # 异常时默认不要求放量确认（安全优先）

        # 原有策略保留（保持原功能不变）
        if (all(df['pctChg'].tail(3) > 0) and df['pctChg'].tail(3).sum() <= 5 and winner_rate > 62):
            signal_tags.append("🔴温和吸筹"); priority = 60; action = "BUY (低吸)"

        # 获取主力资金净流入（用于激进信号过滤，单位：元）
        main_force_inflow = 0
        try:
            main_force_inflow = self.get_main_force_net_inflow(code)
        except Exception:
            pass  # 获取失败时不影响其他逻辑，默认为0
        
        if all(df['turn'].tail(2) > 5) and winner_rate > 70:
            # 激进信号：🔥换手锁仓 - 需要主力资金净流入 > 1000万元（10000000元）
            if main_force_inflow > 10000000:
                signal_tags.append("🔥换手锁仓"); priority = max(priority, 70); action = "BUY (博弈)"

        # 激进信号：🐲妖股基因 - 需要放量确认 + 主力资金净流入 > 1000万元（10000000元）
        if len(df.tail(60)[df.tail(60)['pctChg'] > 9.5]) >= 3 and winner_rate > 80:
            if has_volume_confirmation and main_force_inflow > 10000000:
                signal_tags.append("🐲妖股基因"); priority = 90; action = "STRONG BUY"

        recent_20 = df.tail(20)
        has_limit_up_20 = len(recent_20[recent_20['pctChg'] > 9.5]) > 0
        is_double_vol = (curr['volume'] > prev['volume'] * 1.8)
        # 激进信号：👑四星共振 - 需要放量确认 + 主力资金净流入 > 1000万元（10000000元）
        if has_limit_up_20 and is_double_vol:
            if has_volume_confirmation and main_force_inflow > 10000000:
                signal_tags.append("👑四星共振"); priority = 100; action = "STRONG BUY"
        
        if rsi is not None and len(df) >= 2:
            prev_rsi = self.calc_rsi(df.iloc[:-1])
            if prev_rsi is not None and prev_rsi < 30 and rsi > 35:
                signal_tags.append("💎RSI超卖反弹")
                priority = max(priority, 65)
                if action in ["WAIT (观望)", "HOLD (持有)"]:
                    action = "BUY (低吸)"
        
        if bb_upper is not None and bb_lower is not None:
            if curr['close'] > bb_upper and curr['volume'] > df['volume'].tail(20).mean() * 1.2:
                signal_tags.append("📊布林带突破")
                priority = max(priority, 75)
                if action in ["WAIT (观望)", "HOLD (持有)"]:
                    action = "BUY (博弈)"
        
        if k is not None and d is not None and len(df) >= 2:
            prev_k, prev_d, _ = self.calc_kdj(df.iloc[:-1])
            if prev_k is not None and prev_d is not None:
                if prev_k <= prev_d and k > d and rsi is not None and rsi > 50:
                    signal_tags.append("🎯KDJ金叉")
                    priority = max(priority, 70)
                    if action in ["WAIT (观望)", "HOLD (持有)"]:
                        action = "BUY (博弈)"
        
        if len(df) >= 200 and not pd.isna(df['MA200'].iloc[-1]):
            ma200_current = df['MA200'].iloc[-1]
            ma200_prev = df['MA200'].iloc[-2] if len(df) >= 201 else ma200_current
            if curr['close'] > ma200_current and ma200_current > ma200_prev:
                signal_tags.append("📉200日均线趋势")
                priority = max(priority, 80)
                if action in ["WAIT (观望)", "HOLD (持有)", "BUY (低吸)"]:
                    action = "BUY (低吸)" if action == "WAIT (观望)" else action

        if prev['close'] > prev['open'] and curr['close'] > prev['close']:
            signal_tags.append("📈多头排列")
            priority = max(priority, 50)
            if action == "WAIT (观望)":
                action = "HOLD (持有)"

        # MACD强势信号
        if self.check_macd_signal(df):
            signal_tags.append("📊 MACD强势")
            priority = max(priority, 80)
            if action in ["WAIT (观望)", "HOLD (持有)", "BUY (低吸)"]:
                action = "BUY (博弈)" if action == "WAIT (观望)" else action

        # 成交量异动信号
        if self.check_volume_anomaly(df):
            signal_tags.append("💥 量能异动")
            priority = max(priority, 75)
            if action in ["WAIT (观望)", "HOLD (持有)"]:
                action = "BUY (博弈)"

        if priority == 0:
            return None

        # 现价展示逻辑（保持原功能不变）
        display_price = curr['close']
        if price_map is not None and code in price_map:
            cached_price = price_map[code]
            if cached_price is not None and cached_price > 0:
                price_diff_ratio = abs(cached_price - curr['close']) / curr['close'] if curr['close'] > 0 else 1.0
                if price_diff_ratio <= 0.20:
                    display_price = cached_price

        if allow_realtime_price and display_price == curr['close'] and (price_map is None or code not in price_map):
            try:
                current_realtime_price = self.get_current_price(
                    code,
                    realtime_data_cache=realtime_data_cache,
                    bs_already_logged_in=True
                )
                if current_realtime_price is not None and current_realtime_price > 0:
                    price_diff_ratio = abs(current_realtime_price - curr['close']) / curr['close'] if curr['close'] > 0 else 1.0
                    if price_diff_ratio <= 0.20:
                        display_price = current_realtime_price
            except:
                pass

        return {
            "priority": priority,
            "action": action,
            "signals": " + ".join(signal_tags),
            "winner_rate": winner_rate,
            "risk_level": risk_level,
            "display_price": display_price,
            "pct_chg": f"{curr['pctChg']:.2f}%"
        }

    def _get_basic_info_cached(self, code):
        """获取股票基本信息（带缓存，避免重复IO）"""
        if code in self._basic_info_cache:
            return self._basic_info_cache[code]
        name, industry, ipo_date = code, "-", "2000-01-01"
        try:
            rs_info = bs.query_stock_basic(code=code)
            if rs_info.next():
                row = rs_info.get_row_data()
                name = row[1]
                ipo_date = row[2]
            rs_ind = bs.query_stock_industry(code)
            if rs_ind.next():
                industry = rs_ind.get_row_data()[3]
        except:
            pass
        self._basic_info_cache[code] = (name, industry, ipo_date)
        return name, industry, ipo_date

    def scan_market_optimized(self, code_list, max_price=None):
        """优化后的市场扫描方法
        
        优化说明：
        1. 在扫描前预处理代码格式，建立价格映射表（如果实时数据可用）
        2. 减少重复的代码格式化和匹配操作
        3. 保持原有功能和进度显示逻辑不变
        4. 新增：优先扫描热门概念板块股票
        """
        # === 应用热点板块过滤 ===
        try:
            hot_concept_stocks = self.get_hot_concepts(top_n=8)
            if hot_concept_stocks and len(hot_concept_stocks) > 0:
                original_set = set(code_list)
                hot_set = set()
                for concept in hot_concept_stocks:
                    hot_set.update(self.get_stocks_in_concept(concept))
                # 标准化 hot_set 为 clean_code 格式
                hot_set_clean = {self.clean_code(c) for c in hot_set}
                filtered_list = list(original_set & hot_set_clean)
                if filtered_list:
                    code_list = filtered_list
                    st.info(f"🔥 已过滤到热门概念板块股票：{len(code_list)} 只")
        except Exception:
            # 网络请求失败时自动回退到全市场扫描
            pass
        # =======================
        
        # 保持原有的进度条逻辑，增加命中数量显示，优化进度显示
        results, alerts, valid_codes_list = [], [], []
        if not self.safe_bs_login():
            st.error("❌ baostock登录失败，无法进行扫描")
            return [], [], []
        total = len(code_list)
        progress_bar = st.progress(0, text=f"🚀 正在扫描 (0/{total}) | 命中: 0 只")
        
        # 在扫描开始时，尝试获取一次实时行情数据（用于优化扫描过程中的价格获取）
        # 增加超时保护，避免第三方行情接口卡死导致整体扫描长时间停滞
        realtime_data_cache = None
        price_map = {}  # 代码到价格的映射表，用于快速查找

        def _fetch_spot_em_with_timeout(timeout_seconds=6):
            try:
                with ThreadPoolExecutor(max_workers=1) as tmp_exec:
                    fut = tmp_exec.submit(ak.stock_zh_a_spot_em)
                    return fut.result(timeout=timeout_seconds)
            except Exception:
                return None
        
        try:
            realtime_data_cache = _fetch_spot_em_with_timeout()
            # 如果成功获取实时数据，使用快速方法建立价格映射表
            if realtime_data_cache is not None and not realtime_data_cache.empty:
                code_column, price_column = self._detect_realtime_columns(realtime_data_cache)
                if code_column and price_column:
                    # 使用快速方法建立价格映射
                    price_map = self._build_price_map_fast(code_list, realtime_data_cache, code_column, price_column)
        except Exception:
            # 如果获取失败，继续使用历史数据，不影响扫描
            pass
        
        # 根据总数决定更新频率
        if total <= 100:
            update_interval = 1  # 少于100个，每个都更新
        elif total <= 500:
            update_interval = 5  # 100-500个，每5个更新一次
        else:
            update_interval = 10  # 500个以上，每10个更新一次
        
        # 针对短期交易：如果扫描时间可能较长，考虑刷新实时数据
        # 刷新策略：每处理100只股票或扫描时间超过1分钟时刷新一次（提高实时性）
        cache_refresh_interval = 100  # 每100只股票刷新一次缓存（缩短间隔，提高实时性）
        last_cache_refresh_time = datetime.datetime.now()  # 使用datetime模块的datetime类
        
        # 并发策略（方案B）：主线程串行拉取历史数据（baostock更稳定），线程池并行做指标计算（CPU更吃）
        # 目标：在不引入接口不稳定风险的前提下，将500只从10+分钟压到约3~6分钟区间
        max_workers = min(12, (os.cpu_count() or 4) * 2)
        max_pending_futures = max_workers * 4  # 控制队列长度，避免内存堆积并让“命中”尽快产出

        # 预先计算日期范围（避免每只股票重复计算，减少小开销）
        end_local = datetime.datetime.now().strftime("%Y-%m-%d")
        start_local = (datetime.datetime.now() - datetime.timedelta(days=150)).strftime("%Y-%m-%d")

        def fetch_history_rows(stock_code):
            """拉取单只股票历史数据（网络IO，保持串行更稳）"""
            stock_code = self.clean_code(stock_code)
            rows = []
            rs = bs.query_history_k_data_plus(
                stock_code,
                "date,open,close,high,low,volume,pctChg,turn",
                start_date=start_local,
                end_date=end_local,
                frequency="d",
                adjustflag="3"
            )
            while rs.next():
                rows.append(rs.get_row_data())
            return stock_code, rows

        def _consume_done_futures(future_map, max_to_consume=None):
            """消费已完成的future，把命中结果写入results/alerts/valid_codes_list（保持原功能不变）"""
            if not future_map:
                return 0

            done, _not_done = wait(set(future_map.keys()), timeout=0, return_when=FIRST_COMPLETED)
            consumed = 0
            for fut in list(done):
                stock_code = future_map.pop(fut, None)
                if stock_code is None:
                    continue
                try:
                    analysis = fut.result()
                except Exception:
                    analysis = None

                if analysis:
                    name, industry, _ipo = self._get_basic_info_cached(stock_code)
                    if self.is_valid(stock_code, name):
                        # 获取主力净流入
                        main_force_inflow = None
                        try:
                            main_force_inflow = self.get_main_force_net_inflow(stock_code)
                        except Exception:
                            main_force_inflow = None
                        
                        # 格式化主力净流入显示：使用 pd.isna() 检查是否为 None 或 NaN
                        if pd.isna(main_force_inflow) or main_force_inflow <= 0:
                            main_force_display = "-"
                        else:
                            main_force_display = f"{main_force_inflow/10000:.1f}"
                        
                        results.append({
                            "代码": stock_code,
                            "名称": name,
                            "所属行业": industry,
                            "现价": analysis["display_price"],
                            "涨跌": analysis["pct_chg"],
                            "获利筹码": analysis["winner_rate"],
                            "风险评级": analysis["risk_level"],
                            "策略信号": analysis["signals"],
                            "主力净流入(万)": main_force_display,
                            "综合评级": analysis["action"],
                            "priority": analysis["priority"]
                        })
                        if analysis["priority"] >= 90:
                            alerts.append(f"{name}")
                        valid_codes_list.append(f"{stock_code} | {name}")

                consumed += 1
                if max_to_consume is not None and consumed >= max_to_consume:
                    break
            return consumed

        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {}
            for code in code_list:
                try:
                    stock_code, rows = fetch_history_rows(code)
                except Exception:
                    completed += 1
                    continue
                # 把“计算部分”丢到线程池并发执行
                fut = executor.submit(
                    self._analyze_single_stock_from_history,
                    stock_code,
                    rows,
                    max_price,
                    realtime_data_cache,
                    price_map,
                    False  # 批量扫描模式下不逐票拉实时价，避免大量外部连接
                )
                future_map[fut] = stock_code

                # 流水线：边提交边消费已完成任务，让“命中”在扫描过程中就持续产出
                _consume_done_futures(future_map, max_to_consume=2)

                # 限制pending队列长度，避免内存堆积；必要时阻塞等待一些任务完成
                while len(future_map) >= max_pending_futures:
                    # 阻塞等至少一个完成
                    wait(set(future_map.keys()), timeout=0.5, return_when=FIRST_COMPLETED)
                    _consume_done_futures(future_map, max_to_consume=10)

                completed += 1
                if completed % update_interval == 0 or completed == total:
                    hit_count = len(results)
                    progress_bar.progress(completed / total, text=f"🔍 扫描中: {stock_code} ({completed}/{total}) | 命中: {hit_count} 只")

            # 收尾：等待剩余future完成（仅优化进度展示，不改变扫描/策略结果）
            remaining_total = len(future_map)
            done_tail = 0
            for fut in as_completed(list(future_map.keys())):
                stock_code = future_map.get(fut)
                try:
                    analysis = fut.result()
                except Exception:
                    analysis = None
                if analysis:
                    name, industry, _ipo = self._get_basic_info_cached(stock_code)
                    if self.is_valid(stock_code, name):
                        # 获取主力净流入
                        main_force_inflow = None
                        try:
                            main_force_inflow = self.get_main_force_net_inflow(stock_code)
                        except Exception:
                            main_force_inflow = None
                        
                        # 格式化主力净流入显示：使用 pd.isna() 检查是否为 None 或 NaN
                        if pd.isna(main_force_inflow) or main_force_inflow <= 0:
                            main_force_display = "-"
                        else:
                            main_force_display = f"{main_force_inflow/10000:.1f}"
                        
                        results.append({
                            "代码": stock_code,
                            "名称": name,
                            "所属行业": industry,
                            "现价": analysis["display_price"],
                            "涨跌": analysis["pct_chg"],
                            "获利筹码": analysis["winner_rate"],
                            "风险评级": analysis["risk_level"],
                            "策略信号": analysis["signals"],
                            "主力净流入(万)": main_force_display,
                            "综合评级": analysis["action"],
                            "priority": analysis["priority"]
                        })
                        if analysis["priority"] >= 90:
                            alerts.append(f"{name}")
                        valid_codes_list.append(f"{stock_code} | {name}")

                done_tail += 1
                # 进度=总数 - 剩余future（失败/跳过的会自然计入已完成），避免出现“500/500但还在算”的错觉
                remaining_now = max(remaining_total - done_tail, 0)
                processed_now = total - remaining_now
                if done_tail % (update_interval * 2) == 0 or remaining_now == 0:
                    hit_count = len(results)
                    progress_bar.progress(min(processed_now / total, 1.0), text=f"🧮 计算收尾: {stock_code} ({min(processed_now, total)}/{total}) | 命中: {hit_count} 只")
                    time.sleep(0.01)

        bs.logout()
        # 显示完成状态，延迟一下再清除，让用户看到完成
        progress_bar.progress(1.0, text=f"✅ 扫描完成！共命中 {len(results)} 只")
        time.sleep(0.5)  # 显示完成状态0.5秒
        progress_bar.empty()
        return results, alerts, valid_codes_list

    def _detect_realtime_columns(self, df_realtime):
        """检测实时行情数据的列名（带缓存机制）
        
        Args:
            df_realtime: 实时行情DataFrame
            
        Returns:
            tuple: (code_column, price_column) 或 (None, None)
        """
        # 如果已经检测过且缓存有效，直接返回
        if self._realtime_columns_checked and self._realtime_code_column and self._realtime_price_column:
            # 验证缓存的列名是否仍然存在
            if (self._realtime_code_column in df_realtime.columns and 
                self._realtime_price_column in df_realtime.columns):
                return self._realtime_code_column, self._realtime_price_column
        
        # 检测代码列
        code_column = None
        for possible_code_col in ['代码', 'code', '股票代码', 'stock_code', '证券代码', 'symbol']:
            if possible_code_col in df_realtime.columns:
                code_column = possible_code_col
                break
        
        # 检测价格列
        price_column = None
        for possible_price_col in ['最新价', 'current_price', '现价', 'price', '最新', 'current', '最新价格']:
            if possible_price_col in df_realtime.columns:
                price_column = possible_price_col
                break
        
        # 缓存检测结果
        if code_column and price_column:
            self._realtime_code_column = code_column
            self._realtime_price_column = price_column
            self._realtime_columns_checked = True
        
        return code_column, price_column
    
    def _normalize_stock_code(self, code):
        """标准化股票代码为6位数字格式（用于匹配akshare数据）
        
        Args:
            code: 股票代码（可能是 'sh.600000', '600000', 'sz.000001' 等格式）
            
        Returns:
            str: 标准化后的6位数字代码
        """
        # 去除前缀
        code_clean = str(code).replace('sh.', '').replace('sz.', '').strip()
        
        # 确保是6位数字格式
        if code_clean.isdigit():
            if len(code_clean) < 6:
                code_clean = code_clean.zfill(6)
            elif len(code_clean) > 6:
                code_clean = code_clean[-6:]
        
        return code_clean
    
    def _build_price_map_fast(self, code_list, realtime_df, code_col, price_col):
        """快速构建价格映射表（优化版本，替代循环匹配）
        
        Args:
            code_list: 待匹配的股票代码列表
            realtime_df: 实时行情DataFrame
            code_col: 代码列名
            price_col: 价格列名
            
        Returns:
            dict: {原始代码（clean后）: 价格} 字典
        """
        if realtime_df is None or realtime_df.empty or code_col not in realtime_df.columns or price_col not in realtime_df.columns:
            return {}
        
        price_map = {}
        
        try:
            # 标准化实时数据中的代码为6位纯数字
            code_series = realtime_df[code_col].astype(str).str.strip()
            # 去除字母、点号等，只保留数字，并补零到6位
            normalized_codes = (
                code_series
                .str.replace('sh', '', regex=False)
                .str.replace('sz', '', regex=False)
                .str.replace('.', '', regex=False)
                .str.replace(r'[^0-9]', '', regex=True)
                .str.strip()
            )
            # 补零到6位
            normalized_codes = normalized_codes.apply(lambda x: x.zfill(6) if x.isdigit() and len(x) <= 6 else (x[-6:] if x.isdigit() and len(x) > 6 else ''))
            
            # 构建标准化代码到价格的映射字典
            normalized_price_map = {}
            for idx, norm_code in enumerate(normalized_codes):
                if norm_code and norm_code.isdigit() and len(norm_code) == 6:
                    try:
                        price = float(realtime_df.iloc[idx][price_col])
                        if price > 0 and price < 1e10:
                            # 如果同一个标准化代码出现多次，保留第一个有效价格
                            if norm_code not in normalized_price_map:
                                normalized_price_map[norm_code] = price
                    except (ValueError, KeyError, IndexError):
                        pass
            
            # 对code_list中每个代码进行匹配
            for code in code_list:
                clean_code = self.clean_code(code)
                target_code = self._normalize_stock_code(clean_code)
                
                # 从映射中查找价格
                if target_code in normalized_price_map:
                    price_map[clean_code] = normalized_price_map[target_code]
        except Exception:
            pass
        
        return price_map
    
    def _get_price_from_dataframe(self, df_realtime, target_code, clean_code):
        """从DataFrame中提取价格（通用方法，支持多种数据源格式）
        
        Args:
            df_realtime: 实时行情DataFrame
            target_code: 标准化后的6位代码
            clean_code: 清理后的代码（带前缀）
            
        Returns:
            float: 价格，如果未找到则返回None
        """
        if df_realtime is None or df_realtime.empty:
            return None

        # 使用缓存的列名检测方法
        code_column, price_column = self._detect_realtime_columns(df_realtime)
        if code_column is None or price_column is None:
            return None

        # 优化后的匹配逻辑：使用pandas向量化操作，按优先级依次尝试匹配
        code_series = df_realtime[code_column].astype(str).str.strip()

        # 策略1: 精确匹配（标准6位代码，最常见情况，优先处理）
        mask = code_series == target_code
        if not mask.any():
            # 策略2: 去除前缀后匹配（处理 'sh600000'、'sz000001' 等格式）
            code_normalized = (
                code_series
                .str.replace('sh', '', regex=False)
                .str.replace('sz', '', regex=False)
                .str.replace('.', '', regex=False)
                .str.strip()
            )
            mask = code_normalized == target_code
            if not mask.any() and target_code.isdigit():
                # 策略3: 去除前导零匹配（处理 '1' 匹配 '000001' 的情况）
                target_no_zero = target_code.lstrip('0')
                if target_no_zero and len(target_no_zero) >= 1:
                    mask = code_normalized == target_no_zero
                # 策略4: 包含匹配（最后备选，性能较低，仅在前三种都失败时使用）
                if not mask.any():
                    mask = code_series.str.contains(target_code, na=False, regex=False)

        # 如果找到匹配，提取价格并验证
        if mask.any():
            matched_row = df_realtime[mask].iloc[0]
            try:
                realtime_price = float(matched_row[price_column])
                # 验证价格是否合理（大于0，且不是异常溢出值）
                if realtime_price > 0 and realtime_price < 1e10:
                    return realtime_price
            except (ValueError, KeyError, IndexError):
                pass

        return None
    
    def _try_akshare_spot_em(self, target_code, clean_code, realtime_data_cache=None):
        """尝试从akshare东方财富实时行情获取价格
        
        Args:
            target_code: 标准化后的6位代码
            clean_code: 清理后的代码
            realtime_data_cache: 可选的缓存数据
            
        Returns:
            float: 价格，如果失败则返回None
        """
        try:
            df_realtime = realtime_data_cache if realtime_data_cache is not None else ak.stock_zh_a_spot_em()
            return self._get_price_from_dataframe(df_realtime, target_code, clean_code)
        except Exception:
            return None
    
    def _try_akshare_spot(self, target_code, clean_code):
        """尝试从akshare实时行情获取价格（备选数据源1）
        
        使用akshare的其他实时行情接口作为备选
        
        Args:
            target_code: 标准化后的6位代码
            clean_code: 清理后的代码
            
        Returns:
            float: 价格，如果失败则返回None
        """
        try:
            # 方法1：尝试使用akshare的实时行情接口（全市场）
            df_realtime = ak.stock_zh_a_spot()
            if df_realtime is not None and not df_realtime.empty:
                price = self._get_price_from_dataframe(df_realtime, target_code, clean_code)
                if price is not None:
                    return price
        except Exception:
            pass
        
        # 方法2：尝试使用akshare的腾讯实时行情接口
        try:
            # 转换代码格式：sh.600000 -> sh600000, sz.000001 -> sz000001
            if clean_code.startswith('sh.'):
                symbol = f"sh{target_code}"
            elif clean_code.startswith('sz.'):
                symbol = f"sz{target_code}"
            else:
                symbol = target_code
            
            # 使用腾讯实时行情接口
            df_realtime = ak.stock_zh_a_spot_qq(symbol=symbol)
            if df_realtime is not None and not df_realtime.empty:
                price = self._get_price_from_dataframe(df_realtime, target_code, clean_code)
                if price is not None:
                    return price
        except Exception:
            pass
        
        return None
    
    def _try_akshare_spot_sina(self, target_code, clean_code):
        """尝试从akshare新浪实时行情获取价格（备选数据源2）
        
        Args:
            target_code: 标准化后的6位代码
            clean_code: 清理后的代码
            
        Returns:
            float: 价格，如果失败则返回None
        """
        try:
            # 转换代码格式：sh.600000 -> sh600000, sz.000001 -> sz000001
            if clean_code.startswith('sh.'):
                symbol = f"sh{target_code}"
            elif clean_code.startswith('sz.'):
                symbol = f"sz{target_code}"
            else:
                symbol = target_code
            
            # 方法1：尝试使用akshare的新浪实时行情接口（全市场）
            try:
                df_realtime = ak.stock_zh_a_spot_sina()
                if df_realtime is not None and not df_realtime.empty:
                    price = self._get_price_from_dataframe(df_realtime, target_code, clean_code)
                    if price is not None:
                        return price
            except Exception:
                pass
            
            # 方法2：尝试使用单股票接口（如果全市场接口失败）
            try:
                df_realtime = ak.stock_zh_a_spot_sina(symbol=symbol)
                if df_realtime is not None and not df_realtime.empty:
                    price = self._get_price_from_dataframe(df_realtime, target_code, clean_code)
                    if price is not None:
                        return price
            except Exception:
                pass
        except Exception:
            pass
        
        return None
    
    def get_current_price(self, code, realtime_data_cache=None, bs_already_logged_in=False):
        """获取股票当前价格 (多数据源方案，提高实时性)
        
        优化说明（针对短期交易，解决价格不实时的问题）：
        1. 多数据源按优先级尝试：akshare东方财富 -> akshare实时 -> akshare新浪 -> baostock
        2. 使用列名缓存，避免重复检测
        3. 简化代码匹配逻辑，使用更高效的pandas操作
        4. 优化异常处理，减少不必要的开销
        5. 增加价格合理性验证，过滤异常值
        
        Args:
            code: 股票代码
            realtime_data_cache: 可选的实时行情数据缓存（DataFrame），用于优化扫描性能
            bs_already_logged_in: Baostock是否已经登录（扫描过程中为True，避免重复登录）
            
        Returns:
            float: 实时价格，如果获取失败则返回None
        """
        clean_code = self.clean_code(code)
        target_code = self._normalize_stock_code(clean_code)
        
        # 策略1：优先使用akshare东方财富实时行情（最常用，支持缓存）
        price = self._try_akshare_spot_em(target_code, clean_code, realtime_data_cache)
        if price is not None:
            return price
        
        # 策略2：尝试akshare实时行情（备选数据源1）
        price = self._try_akshare_spot(target_code, clean_code)
        if price is not None:
            return price
        
        # 策略3：尝试akshare新浪实时行情（备选数据源2）
        price = self._try_akshare_spot_sina(target_code, clean_code)
        if price is not None:
            return price
        
        # 如果akshare失败，或者未找到数据，则回退到Baostock获取最新收盘价
        # 注意：对于短期交易，收盘价可能不是最新价格，但作为备用方案
        # 在扫描过程中（bs_already_logged_in为True），直接使用已登录的baostock，避免重复登录
        try:
            if not bs_already_logged_in:
                bs.login()
            
            end = datetime.datetime.now().strftime("%Y-%m-%d")
            # 尝试获取当天数据，如果失败则回溯几天（最多回溯5天）
            for i in range(5):
                start = (datetime.datetime.now() - datetime.timedelta(days=i)).strftime("%Y-%m-%d")
                rs = bs.query_history_k_data_plus(clean_code, "date,close", start_date=start, end_date=end, frequency="d", adjustflag="3")
                data = []
                while rs.next(): 
                    data.append(rs.get_row_data())
                if data:
                    baostock_price = float(data[-1][1])
                    # 验证价格合理性
                    if baostock_price > 0 and baostock_price < 1e10:
                        if not bs_already_logged_in:
                            bs.logout()
                        return baostock_price  # 返回最新收盘价
            
            if not bs_already_logged_in:
                bs.logout()
            return None
        except Exception:
            if not bs_already_logged_in:
                try:
                    bs.logout()
                except Exception:
                    pass
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
            if not self.safe_bs_login():
                return None
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
        """技术面趋势推演：根据技术指标判断未来趋势方向"""
        if df is None or len(df) < 30: return None
        try:
            # 计算当前价格
            current_price = df['close'].iloc[-1]
            
            # 计算技术指标
            df['MA5'] = df['close'].rolling(5).mean()
            df['MA20'] = df['close'].rolling(20).mean()
            rsi = self.calc_rsi(df)
            
            # 计算近3日涨跌幅
            if len(df) >= 3:
                recent_3d_change = ((df['close'].iloc[-1] - df['close'].iloc[-4]) / df['close'].iloc[-4]) * 100
            else:
                recent_3d_change = 0
            
            # 获取当前MA5和MA20值
            ma5_current = df['MA5'].iloc[-1] if not pd.isna(df['MA5'].iloc[-1]) else None
            ma20_current = df['MA20'].iloc[-1] if not pd.isna(df['MA20'].iloc[-1]) else None
            
            # 判断趋势
            trend = "震荡"
            color = "blue"
            title = "📊 技术推演：震荡趋势"
            desc = "技术指标显示当前处于震荡整理状态"
            action = "建议持有观望，等待明确方向"
            
            # 上涨趋势条件：RSI > 50 + MA5 > MA20 + 近3日涨幅 > 5%
            if (rsi is not None and rsi > 50 and 
                ma5_current is not None and ma20_current is not None and ma5_current > ma20_current and
                recent_3d_change > 5):
                trend = "上涨"
                color = "red"
                title = "📊 技术推演：上涨趋势"
                desc = f"RSI处于强势区间({rsi:.1f})，均线多头排列，近3日涨幅{recent_3d_change:.2f}%，技术面偏强"
                action = "建议持有或逢低买入，关注突破信号"
            
            # 下跌趋势条件：RSI < 40 + MA5 < MA20 + 近3日跌幅 > 3%
            elif (rsi is not None and rsi < 40 and 
                  ma5_current is not None and ma20_current is not None and ma5_current < ma20_current and
                  recent_3d_change < -3):
                trend = "下跌"
                color = "green"
                title = "📊 技术推演：下跌趋势"
                desc = f"RSI处于弱势区间({rsi:.1f})，均线空头排列，近3日跌幅{abs(recent_3d_change):.2f}%，技术面偏弱"
                action = "建议谨慎观望或减仓，注意风险控制"
            
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
            
            # prices 和 changes 设为当前价格和0（保持输出结构不变）
            pred_prices = [current_price, current_price, current_price]
            changes = [0, 0, 0]

            return {
                "dates": dates,
                "prices": pred_prices,
                "changes": changes,
                "pred_price": current_price,
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
if 'watchlist' not in st.session_state: st.session_state['watchlist'] = []

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

# 我的关注列表功能
st.sidebar.markdown("---")
st.sidebar.subheader("⭐ 我的关注")

# 显示关注列表
if st.session_state.get('watchlist'):
    watchlist_count = len(st.session_state['watchlist'])
    st.sidebar.info(f"📋 已关注 {watchlist_count} 只股票")
    
    # 更新按钮
    if st.sidebar.button("🔄 更新价格和资金", key="update_watchlist"):
        with st.sidebar.spinner("正在更新..."):
            for item in st.session_state['watchlist']:
                code = item.get('代码', '')
                if code:
                    try:
                        # 更新当前价格
                        current_price = engine.get_current_price(code)
                        if current_price:
                            item['当前价格'] = f"{current_price:.2f}"
                        # 更新主力净流入
                        main_force = engine.get_main_force_net_inflow(code)
                        if main_force and not pd.isna(main_force) and main_force > 0:
                            item['主力净流入(万)'] = f"{main_force/10000:.1f}"
                        else:
                            item['主力净流入(万)'] = "-"
                    except Exception:
                        pass
            st.sidebar.success("✅ 更新完成")
    
    # 显示关注列表
    for i, item in enumerate(st.session_state['watchlist']):
        code = item.get('代码', 'N/A')
        name = item.get('名称', 'N/A')
        with st.sidebar.expander(f"{code} | {name}", expanded=False):
            st.write(f"**代码**: {code}")
            st.write(f"**名称**: {name}")
            st.write(f"**当前价格**: {item.get('当前价格', '未更新')}")
            st.write(f"**主力净流入**: {item.get('主力净流入(万)', 'N/A')}")
            st.write(f"**策略信号**: {item.get('策略信号', 'N/A')}")
            st.write(f"**综合评级**: {item.get('综合评级', 'N/A')}")
            st.write(f"**添加时间**: {item.get('添加时间', 'N/A')}")
            if st.button("🗑️ 移除", key=f"remove_watch_{i}"):
                st.session_state['watchlist'].pop(i)
                st.rerun()
else:
    st.sidebar.info("💡 暂无关注股票，在扫描结果中点击 ⭐ 关注 按钮添加")

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
show_watchlist = False

if check_admin_access():
    main_tabs = st.tabs(["📊 量化分析", "⭐ 我的关注", "👨‍💼 管理后台"])
    if main_tabs[1]:  # 如果点击了"我的关注"标签
        show_watchlist = True
    elif main_tabs[2]:  # 如果点击了管理后台标签
        show_admin = True
else:
    main_tabs = st.tabs(["📊 量化分析", "⭐ 我的关注"])
    if main_tabs[1]:  # 如果点击了"我的关注"标签
        show_watchlist = True

# 根据选择的标签页显示内容
if show_watchlist:
    # ==========================================
    # 我的关注页面
    # ==========================================
    st.title("⭐ 我的关注列表")
    st.caption("管理您关注的股票，实时查看价格和资金流向")
    
    if st.session_state.get('watchlist'):
        watchlist_count = len(st.session_state['watchlist'])
        st.success(f"📋 您已关注 {watchlist_count} 只股票")
        
        # 更新按钮
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🔄 更新所有数据", type="primary"):
                with st.spinner("正在更新价格和资金流向..."):
                    for item in st.session_state['watchlist']:
                        code = item.get('代码', '')
                        if code:
                            try:
                                # 更新当前价格
                                current_price = engine.get_current_price(code)
                                if current_price:
                                    item['当前价格'] = f"{current_price:.2f}"
                                # 更新主力净流入
                                main_force = engine.get_main_force_net_inflow(code)
                                if main_force and not pd.isna(main_force) and main_force > 0:
                                    item['主力净流入(万)'] = f"{main_force/10000:.1f}"
                                else:
                                    item['主力净流入(万)'] = "-"
                            except Exception:
                                pass
                    st.success("✅ 更新完成")
                    st.rerun()
        
        # 显示关注列表表格
        watchlist_data = []
        for item in st.session_state['watchlist']:
            watchlist_data.append({
                '代码': item.get('代码', 'N/A'),
                '名称': item.get('名称', 'N/A'),
                '当前价格': item.get('当前价格', '未更新'),
                '主力净流入(万)': item.get('主力净流入(万)', 'N/A'),
                '策略信号': item.get('策略信号', 'N/A'),
                '综合评级': item.get('综合评级', 'N/A'),
                '添加时间': item.get('添加时间', 'N/A')
            })
        
        if watchlist_data:
            df_watchlist = pd.DataFrame(watchlist_data)
            st.dataframe(df_watchlist, hide_index=True, use_container_width=True)
        
        # 移除按钮
        st.markdown("---")
        st.markdown("### 🗑️ 移除关注")
        for i, item in enumerate(st.session_state['watchlist']):
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                st.write(f"**{item.get('代码', 'N/A')}** | {item.get('名称', 'N/A')}")
            with col2:
                st.write(f"主力: {item.get('主力净流入(万)', 'N/A')}")
            with col3:
                if st.button("🗑️ 移除", key=f"remove_watch_main_{i}"):
                    removed_name = item.get('名称', 'N/A')
                    st.session_state['watchlist'].pop(i)
                    st.success(f"✅ 已移除 {removed_name}")
                    st.rerun()
    else:
        st.info("💡 您还没有关注任何股票。在扫描结果中点击 ⭐ 关注 按钮添加股票到关注列表。")
        st.markdown("""
        ### 📝 使用说明：
        1. 在左侧边栏点击 "🚀 启动全策略扫描" 进行股票扫描
        2. 扫描完成后，在扫描结果下方的 "⭐ 快速关注" 区域点击关注按钮
        3. 已关注的股票会显示在 "⭐ 我的关注" 标签页中
        4. 可以随时更新价格和资金流向数据
        """)

elif show_admin:
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
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
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
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
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
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
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
            # Streamlit 默认是“正数绿、负数红”，这里统一用 inverse 反转为“正数红、负数绿”
            if total_profit > 0:
                st.metric("总盈亏", f"¥{total_profit:,.2f}", delta=f"+{total_profit_rate:.2f}%", delta_color="inverse")
            elif total_profit < 0:
                st.metric("总盈亏", f"¥{total_profit:,.2f}", delta=f"{total_profit_rate:.2f}%", delta_color="inverse")
            else:
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
                            
                            # 显示主力净流入
                            main_force_inflow = 0
                            try:
                                main_force_inflow = engine.get_main_force_net_inflow(selected_code)
                            except Exception:
                                pass
                            
                            if main_force_inflow > 0:
                                main_force_display = f"{main_force_inflow/10000:.1f} 万元"
                            else:
                                main_force_display = "暂无数据"
                            
                            st.markdown(f"💰 主力净流入：{main_force_display}")
                            
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
                                    # 统一：红涨绿跌（inverse 反转默认配色）
                                    st.metric(
                                        label=date_label,
                                        value=f"¥{pred_price:.2f}",
                                        delta=f"{change_amount:+.2f} ({change:+.2f}%)",
                                        delta_color="inverse"
                                    )
                                    direction_cn = "上涨" if change_amount >= 0 else "下跌"
                                    st.caption(f"预计较当前{direction_cn} {abs(change_amount):.2f} 元（{change:+.2f}%）")
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

    # 标记主力高控盘标的，方便列表中快速定位（不影响原有数据结构）
    alert_set = set(st.session_state.get('alerts', []) or [])
    df_scan['主力标记'] = df_scan['名称'].apply(lambda x: "🔥" if x in alert_set else "")
    
    # 调整列顺序，确保"主力标记"列显示在最后
    columns_order = [col for col in df_scan.columns if col != '主力标记'] + ['主力标记']
    df_scan = df_scan[columns_order]
    
    # 显示命中股票数量
    total_count = len(df_scan)
    st.success(f"✅ **扫描完成！共命中 {total_count} 只符合条件的股票**")
    
    # 显示主力高控盘标的（priority >= 90的股票）——上方仅显示股票名称，便于阅读
    # 检查 alerts 是否存在且不为空
    alerts = st.session_state.get('alerts', [])
    if alerts and len(alerts) > 0:
        alert_count = len(alerts)
        alert_names = "、".join(alerts)
        st.success(f"🔥 **发现 {alert_count} 只【主力高控盘】标的：{alert_names}**")
    else:
        # 检查是否有 priority >= 90 的股票（从扫描结果中查找）
        high_priority_stocks = df_scan[df_scan['priority'] >= 90]
        if len(high_priority_stocks) > 0:
            # 如果有但 alerts 为空，说明可能是数据同步问题，从结果中提取
            high_priority_names = high_priority_stocks['名称'].tolist()
            alert_count = len(high_priority_names)
            alert_names = "、".join(high_priority_names)
            st.success(f"🔥 **发现 {alert_count} 只【主力高控盘】标的：{alert_names}**")
            # 同步更新 alerts
            st.session_state['alerts'] = high_priority_names
        else:
            # 显示策略说明
            st.info("💡 本次扫描未发现 priority ≥ 90 的【主力高控盘】标的。\n\n"
                   "**触发条件说明：**\n"
                   "- 🐲 **妖股基因**（priority=90）：近60日涨停≥3次 + 获利筹码>80% + 放量确认 + 主力净流入>1000万\n"
                   "- 👑 **四星共振**（priority=100）：近20日有涨停 + 倍量 + 放量确认 + 主力净流入>1000万")
    
    # 配置列提示信息
    column_config = {
        "主力标记": st.column_config.TextColumn("标记", help="主力高控盘标的，用🔥标出"),
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
    
    # 添加关注按钮（在表格下方，使用紧凑布局）
    st.markdown("---")
    st.markdown("### ⭐ 快速关注")
    
    # 使用更紧凑的方式显示关注按钮
    watchlist_codes = {w.get('代码') for w in st.session_state.get('watchlist', [])}
    
    # 每行显示3个按钮
    rows_per_line = 3
    for i in range(0, len(df_scan), rows_per_line):
        cols = st.columns(rows_per_line)
        for j, col in enumerate(cols):
            if i + j < len(df_scan):
                row = df_scan.iloc[i + j]
                code = row['代码']
                name = row['名称']
                is_watched = code in watchlist_codes
                
                with col:
                    if is_watched:
                        st.button("✅ 已关注", key=f"watch_btn_{i+j}", disabled=True, use_container_width=True)
                    else:
                        if st.button(f"⭐ {name[:8]}", key=f"watch_btn_{i+j}", use_container_width=True):
                            # 添加到关注列表
                            watch_item = {
                                '代码': code,
                                '名称': name,
                                '主力净流入(万)': row.get('主力净流入(万)', '-'),
                                '策略信号': row.get('策略信号', '-'),
                                '综合评级': row.get('综合评级', '-'),
                                'priority': row.get('priority', 0),
                                '添加时间': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            st.session_state['watchlist'].append(watch_item)
                            st.success(f"✅ 已添加 {name} 到关注列表")
                            st.rerun()

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
                        
                        # 显示主力净流入
                        main_force_inflow = 0
                        try:
                            main_force_inflow = engine.get_main_force_net_inflow(target_code)
                        except Exception:
                            pass
                        
                        if main_force_inflow > 0:
                            main_force_display = f"{main_force_inflow/10000:.1f} 万元"
                        else:
                            main_force_display = "暂无数据"
                        
                        st.markdown(f"💰 主力净流入：{main_force_display}")
                        
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
                                # 统一配色：使用 inverse，让“涨=红、跌=绿”，箭头方向仍按涨跌变化
                                st.metric(
                                    label=date_label,
                                    value=f"¥{pred_price:.2f}", 
                                    delta=f"{change_amount:+.2f} ({change:+.2f}%)",
                                    delta_color="inverse"
                                )
                                direction_cn = "上涨" if change_amount >= 0 else "下跌"
                                st.caption(f"预计较当前{direction_cn} {abs(change_amount):.2f} 元（{change:+.2f}%）")
                        
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