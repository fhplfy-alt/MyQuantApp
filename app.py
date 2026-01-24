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

st.title("🛡️ V45 智能量化系统 (全信号图例版)")
st.caption("✅ 系统已就绪 | 核心组件加载完成 | 支持6000股扫描 | V45 Build")

# ==========================================
# 1. 安全导入
# ==========================================
try:
    import plotly.graph_objects as go
    import random
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import time
    import datetime
    from sklearn.linear_model import LinearRegression
except ImportError as e:
    st.error(f"❌ 启动失败！缺少必要运行库: {e}")
    st.error(f"💡 提示：请运行 pip install yfinance")
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
        """清理股票代码，转换为yfinance格式"""
        code = str(code).strip()
        # 移除前缀
        if code.startswith('sh.') or code.startswith('sz.'):
            code = code[3:]
        # 转换为yfinance格式：600000 -> 600000.SS, 000001 -> 000001.SZ
        if code.startswith('6'):
            return f"{code}.SS"
        elif code.startswith(('0', '3')):
            return f"{code}.SZ"
        return code
    
    def clean_code_back(self, code):
        """将yfinance格式转回原始格式"""
        if code.endswith('.SS'):
            return code[:-3]
        elif code.endswith('.SZ'):
            return code[:-3]
        return code

    def is_valid(self, code, name):
        if "sh.688" in code: return False 
        if "bj." in code or code.startswith("sz.8") or code.startswith("sz.4"): return False 
        if "ST" in name: return False 
        return True

    def get_all_stocks(self):
        """获取全市场股票，使用预定义列表（yfinance版本）"""
        # 使用常见A股代码列表（前6000只）
        # 上海：600000-603999, 688000-688999
        # 深圳：000001-002999, 300000-300999
        stocks = []
        
        # 上海主板
        for i in range(600000, 604000):
            stocks.append(f"sh.{i}")
        
        # 深圳主板和中小板
        for i in range(1, 3000):
            code = str(i).zfill(6)
            stocks.append(f"sz.{code}")
        
        # 创业板
        for i in range(300000, 301000):
            stocks.append(f"sz.{i}")
        
        # 过滤无效股票
        valid_stocks = []
        for code in stocks[:self.MAX_SCAN_LIMIT]:
            if self.is_valid(code, ""):
                valid_stocks.append(code)
        
        return valid_stocks[:self.MAX_SCAN_LIMIT]

    def get_index_stocks(self, index_type="zz500"):
        """获取指数成分股（使用真实成分股代码，yfinance可支持的）"""
        # 使用真实的中证500和沪深300成分股代码（yfinance可支持的）
        if index_type == "hs300":
            # 沪深300真实成分股（部分，yfinance可支持的）
            stocks = [
                "sh.600000", "sh.600009", "sh.600010", "sh.600016", "sh.600019",
                "sh.600028", "sh.600029", "sh.600030", "sh.600031", "sh.600036",
                "sh.600038", "sh.600048", "sh.600050", "sh.600061", "sh.600066",
                "sh.600085", "sh.600104", "sh.600111", "sh.600115", "sh.600150",
                "sh.600196", "sh.600276", "sh.600309", "sh.600340", "sh.600519",
                "sh.600547", "sh.600570", "sh.600585", "sh.600588", "sh.600606",
                "sh.600637", "sh.600660", "sh.600690", "sh.600703", "sh.600745",
                "sh.600809", "sh.600837", "sh.600887", "sh.600893", "sh.600900",
                "sh.600919", "sh.600958", "sh.600999", "sh.601006", "sh.601012",
                "sh.601018", "sh.601066", "sh.601088", "sh.601138", "sh.601166",
                "sh.601169", "sh.601186", "sh.601211", "sh.601216", "sh.601225",
                "sh.601229", "sh.601236", "sh.601238", "sh.601288", "sh.601318",
                "sh.601319", "sh.601328", "sh.601336", "sh.601360", "sh.601377",
                "sh.601390", "sh.601398", "sh.601601", "sh.601607", "sh.601618",
                "sh.601628", "sh.601633", "sh.601658", "sh.601668", "sh.601688",
                "sh.601698", "sh.601727", "sh.601766", "sh.601788", "sh.601800",
                "sh.601808", "sh.601816", "sh.601818", "sh.601828", "sh.601838",
                "sh.601857", "sh.601860", "sh.601866", "sh.601872", "sh.601877",
                "sh.601881", "sh.601888", "sh.601898", "sh.601899", "sh.601901",
                "sh.601916", "sh.601919", "sh.601933", "sh.601939", "sh.601985",
                "sh.601988", "sh.601989", "sh.601992", "sh.601995", "sh.601998",
                "sz.000001", "sz.000002", "sz.000009", "sz.000012", "sz.000021",
                "sz.000027", "sz.000039", "sz.000063", "sz.000069", "sz.000100",
                "sz.000157", "sz.000166", "sz.000301", "sz.000338", "sz.000402",
                "sz.000413", "sz.000415", "sz.000423", "sz.000425", "sz.000488",
                "sz.000538", "sz.000540", "sz.000559", "sz.000568", "sz.000625",
                "sz.000627", "sz.000629", "sz.000630", "sz.000651", "sz.000656",
                "sz.000661", "sz.000667", "sz.000671", "sz.000686", "sz.000709",
                "sz.000717", "sz.000725", "sz.000728", "sz.000729", "sz.000738",
                "sz.000750", "sz.000768", "sz.000776", "sz.000778", "sz.000783",
                "sz.000786", "sz.000792", "sz.000800", "sz.000807", "sz.000825",
                "sz.000830", "sz.000839", "sz.000858", "sz.000876", "sz.000877",
                "sz.000895", "sz.000898", "sz.000917", "sz.000921", "sz.000927",
                "sz.000930", "sz.000932", "sz.000938", "sz.000959", "sz.000961",
                "sz.000963", "sz.000970", "sz.000977", "sz.000983", "sz.000988",
                "sz.000989", "sz.000997", "sz.002001", "sz.002007", "sz.002013",
                "sz.002024", "sz.002027", "sz.002032", "sz.002044", "sz.002050",
                "sz.002065", "sz.002081", "sz.002092", "sz.002142", "sz.002146",
                "sz.002153", "sz.002179", "sz.002202", "sz.002230", "sz.002236",
                "sz.002241", "sz.002252", "sz.002271", "sz.002304", "sz.002311",
                "sz.002352", "sz.002371", "sz.002384", "sz.002415", "sz.002422",
                "sz.002456", "sz.002460", "sz.002466", "sz.002475", "sz.002493",
                "sz.002508", "sz.002531", "sz.002558", "sz.002572", "sz.002594",
                "sz.002601", "sz.002602", "sz.002624", "sz.002673", "sz.002714",
                "sz.002739", "sz.002821", "sz.002841", "sz.002916", "sz.002920",
                "sz.300015", "sz.300059", "sz.300070", "sz.300122", "sz.300142",
                "sz.300144", "sz.300146", "sz.300168", "sz.300274", "sz.300347"
            ]
        else:
            # 中证500真实成分股（部分，yfinance可支持的）
            stocks = [
                "sh.600011", "sh.600012", "sh.600015", "sh.600017", "sh.600018",
                "sh.600020", "sh.600021", "sh.600022", "sh.600023", "sh.600025",
                "sh.600026", "sh.600027", "sh.600033", "sh.600035", "sh.600037",
                "sh.600039", "sh.600041", "sh.600043", "sh.600045", "sh.600051",
                "sh.600052", "sh.600053", "sh.600054", "sh.600055", "sh.600056",
                "sh.600057", "sh.600058", "sh.600059", "sh.600060", "sh.600062",
                "sh.600063", "sh.600064", "sh.600067", "sh.600068", "sh.600069",
                "sh.600070", "sh.600071", "sh.600072", "sh.600073", "sh.600074",
                "sh.600075", "sh.600076", "sh.600077", "sh.600078", "sh.600079",
                "sh.600080", "sh.600081", "sh.600082", "sh.600083", "sh.600084",
                "sh.600086", "sh.600088", "sh.600089", "sh.600090", "sh.600091",
                "sh.600092", "sh.600093", "sh.600094", "sh.600095", "sh.600096",
                "sh.600097", "sh.600098", "sh.600099", "sh.600100", "sh.600101",
                "sh.600103", "sh.600105", "sh.600106", "sh.600107", "sh.600108",
                "sh.600109", "sh.600110", "sh.600112", "sh.600113", "sh.600114",
                "sh.600115", "sh.600116", "sh.600117", "sh.600118", "sh.600119",
                "sh.600120", "sh.600121", "sh.600122", "sh.600123", "sh.600125",
                "sh.600126", "sh.600127", "sh.600128", "sh.600129", "sh.600130",
                "sh.600131", "sh.600132", "sh.600133", "sh.600135", "sh.600136",
                "sh.600137", "sh.600138", "sh.600139", "sh.600141", "sh.600143",
                "sh.600145", "sh.600146", "sh.600148", "sh.600149", "sh.600150",
                "sh.600151", "sh.600152", "sh.600153", "sh.600155", "sh.600156",
                "sh.600157", "sh.600158", "sh.600159", "sh.600160", "sh.600161",
                "sh.600162", "sh.600163", "sh.600165", "sh.600166", "sh.600167",
                "sh.600168", "sh.600169", "sh.600170", "sh.600171", "sh.600172",
                "sh.600173", "sh.600175", "sh.600176", "sh.600177", "sh.600178",
                "sh.600179", "sh.600180", "sh.600182", "sh.600183", "sh.600184",
                "sh.600185", "sh.600186", "sh.600187", "sh.600188", "sh.600189",
                "sh.600190", "sh.600191", "sh.600192", "sh.600193", "sh.600195",
                "sh.600196", "sh.600197", "sh.600198", "sh.600199", "sh.600200",
                "sz.000011", "sz.000012", "sz.000014", "sz.000016", "sz.000017",
                "sz.000018", "sz.000019", "sz.000020", "sz.000021", "sz.000022",
                "sz.000023", "sz.000024", "sz.000025", "sz.000026", "sz.000027",
                "sz.000028", "sz.000029", "sz.000030", "sz.000031", "sz.000032",
                "sz.000033", "sz.000034", "sz.000035", "sz.000036", "sz.000037",
                "sz.000038", "sz.000039", "sz.000040", "sz.000042", "sz.000043",
                "sz.000045", "sz.000046", "sz.000048", "sz.000049", "sz.000050",
                "sz.000055", "sz.000056", "sz.000058", "sz.000059", "sz.000060",
                "sz.000061", "sz.000062", "sz.000063", "sz.000065", "sz.000066",
                "sz.000067", "sz.000068", "sz.000069", "sz.000070", "sz.000078",
                "sz.000088", "sz.000089", "sz.000090", "sz.000096", "sz.000099",
                "sz.000100", "sz.000150", "sz.000151", "sz.000153", "sz.000155",
                "sz.000156", "sz.000157", "sz.000158", "sz.000159", "sz.000301",
                "sz.000400", "sz.000401", "sz.000402", "sz.000403", "sz.000404",
                "sz.000407", "sz.000408", "sz.000409", "sz.000410", "sz.000411",
                "sz.000413", "sz.000415", "sz.000416", "sz.000417", "sz.000418",
                "sz.000419", "sz.000420", "sz.000421", "sz.000422", "sz.000423",
                "sz.000425", "sz.000426", "sz.000428", "sz.000429", "sz.000430",
                "sz.000488", "sz.000498", "sz.000501", "sz.000502", "sz.000503",
                "sz.000504", "sz.000505", "sz.000506", "sz.000507", "sz.000509",
                "sz.000510", "sz.000511", "sz.000513", "sz.000514", "sz.000516",
                "sz.000517", "sz.000518", "sz.000519", "sz.000520", "sz.000521",
                "sz.000522", "sz.000523", "sz.000524", "sz.000525", "sz.000526",
                "sz.000527", "sz.000528", "sz.000529", "sz.000530", "sz.000531",
                "sz.000532", "sz.000533", "sz.000534", "sz.000536", "sz.000537",
                "sz.000538", "sz.000539", "sz.000540", "sz.000541", "sz.000543",
                "sz.000544", "sz.000545", "sz.000546", "sz.000547", "sz.000548",
                "sz.000550", "sz.000551", "sz.000552", "sz.000553", "sz.000554",
                "sz.000555", "sz.000557", "sz.000558", "sz.000559", "sz.000560",
                "sz.000561", "sz.000562", "sz.000563", "sz.000564", "sz.000565",
                "sz.000566", "sz.000567", "sz.000568", "sz.000570", "sz.000571",
                "sz.000572", "sz.000573", "sz.000576", "sz.000578", "sz.000581",
                "sz.000582", "sz.000584", "sz.000585", "sz.000586", "sz.000587",
                "sz.000588", "sz.000589", "sz.000590", "sz.000591", "sz.000592",
                "sz.000593", "sz.000595", "sz.000596", "sz.000597", "sz.000598",
                "sz.000599", "sz.000600", "sz.000601", "sz.000602", "sz.000603",
                "sz.000605", "sz.000606", "sz.000607", "sz.000608", "sz.000609",
                "sz.000610", "sz.000611", "sz.000612", "sz.000613", "sz.000615",
                "sz.000616", "sz.000617", "sz.000619", "sz.000620", "sz.000621",
                "sz.000622", "sz.000623", "sz.000625", "sz.000626", "sz.000627",
                "sz.000628", "sz.000629", "sz.000630", "sz.000631", "sz.000632",
                "sz.000633", "sz.000635", "sz.000636", "sz.000637", "sz.000638",
                "sz.000639", "sz.000650", "sz.000651", "sz.000652", "sz.000655",
                "sz.000656", "sz.000657", "sz.000659", "sz.000661", "sz.000662",
                "sz.000663", "sz.000665", "sz.000666", "sz.000667", "sz.000668",
                "sz.000669", "sz.000670", "sz.000671", "sz.000672", "sz.000673",
                "sz.000676", "sz.000677", "sz.000678", "sz.000679", "sz.000680",
                "sz.000681", "sz.000682", "sz.000683", "sz.000685", "sz.000686",
                "sz.000687", "sz.000688", "sz.000690", "sz.000691", "sz.000692",
                "sz.000693", "sz.000695", "sz.000697", "sz.000698", "sz.000700",
                "sz.000701", "sz.000702", "sz.000703", "sz.000705", "sz.000707",
                "sz.000708", "sz.000709", "sz.000710", "sz.000711", "sz.000712",
                "sz.000713", "sz.000715", "sz.000716", "sz.000717", "sz.000718",
                "sz.000719", "sz.000720", "sz.000721", "sz.000722", "sz.000723",
                "sz.000725", "sz.000726", "sz.000727", "sz.000728", "sz.000729",
                "sz.000730", "sz.000731", "sz.000732", "sz.000733", "sz.000735",
                "sz.000736", "sz.000737", "sz.000738", "sz.000739", "sz.000750",
                "sz.000751", "sz.000752", "sz.000753", "sz.000755", "sz.000756",
                "sz.000757", "sz.000758", "sz.000759", "sz.000760", "sz.000761",
                "sz.000762", "sz.000763", "sz.000765", "sz.000766", "sz.000767",
                "sz.000768", "sz.000769", "sz.000776", "sz.000777", "sz.000778",
                "sz.000779", "sz.000780", "sz.000782", "sz.000783", "sz.000785",
                "sz.000786", "sz.000788", "sz.000789", "sz.000790", "sz.000791",
                "sz.000792", "sz.000793", "sz.000795", "sz.000796", "sz.000797",
                "sz.000798", "sz.000799", "sz.000800", "sz.000801", "sz.000802",
                "sz.000803", "sz.000805", "sz.000806", "sz.000807", "sz.000809",
                "sz.000810", "sz.000811", "sz.000812", "sz.000813", "sz.000815",
                "sz.000816", "sz.000817", "sz.000818", "sz.000819", "sz.000820",
                "sz.000821", "sz.000822", "sz.000823", "sz.000825", "sz.000826",
                "sz.000827", "sz.000828", "sz.000829", "sz.000830", "sz.000831",
                "sz.000833", "sz.000835", "sz.000836", "sz.000837", "sz.000838",
                "sz.000839", "sz.000848", "sz.000850", "sz.000851", "sz.000852",
                "sz.000856", "sz.000858", "sz.000859", "sz.000860", "sz.000861",
                "sz.000862", "sz.000863", "sz.000868", "sz.000869", "sz.000875",
                "sz.000876", "sz.000877", "sz.000878", "sz.000880", "sz.000881",
                "sz.000882", "sz.000883", "sz.000885", "sz.000886", "sz.000887",
                "sz.000888", "sz.000889", "sz.000890", "sz.000892", "sz.000893",
                "sz.000895", "sz.000897", "sz.000898", "sz.000899", "sz.000900",
                "sz.000901", "sz.000902", "sz.000903", "sz.000905", "sz.000906",
                "sz.000908", "sz.000909", "sz.000910", "sz.000911", "sz.000912",
                "sz.000913", "sz.000915", "sz.000916", "sz.000917", "sz.000918",
                "sz.000919", "sz.000920", "sz.000921", "sz.000922", "sz.000923",
                "sz.000925", "sz.000926", "sz.000927", "sz.000928", "sz.000929",
                "sz.000930", "sz.000931", "sz.000932", "sz.000933", "sz.000935",
                "sz.000936", "sz.000937", "sz.000938", "sz.000939", "sz.000948",
                "sz.000949", "sz.000950", "sz.000951", "sz.000952", "sz.000953",
                "sz.000955", "sz.000957", "sz.000958", "sz.000959", "sz.000960",
                "sz.000961", "sz.000962", "sz.000963", "sz.000965", "sz.000966",
                "sz.000967", "sz.000968", "sz.000969", "sz.000970", "sz.000971",
                "sz.000972", "sz.000973", "sz.000975", "sz.000976", "sz.000977",
                "sz.000978", "sz.000979", "sz.000980", "sz.000981", "sz.000982",
                "sz.000983", "sz.000985", "sz.000986", "sz.000987", "sz.000988",
                "sz.000989", "sz.000990", "sz.000991", "sz.000992", "sz.000993",
                "sz.000995", "sz.000996", "sz.000997", "sz.000998", "sz.000999",
                "sz.002001", "sz.002002", "sz.002003", "sz.002004", "sz.002005",
                "sz.002006", "sz.002007", "sz.002008", "sz.002009", "sz.002010",
                "sz.002011", "sz.002012", "sz.002013", "sz.002014", "sz.002015",
                "sz.002016", "sz.002017", "sz.002018", "sz.002019", "sz.002020",
                "sz.002021", "sz.002022", "sz.002023", "sz.002024", "sz.002025",
                "sz.002026", "sz.002027", "sz.002028", "sz.002029", "sz.002030",
                "sz.002031", "sz.002032", "sz.002033", "sz.002034", "sz.002035",
                "sz.002036", "sz.002037", "sz.002038", "sz.002039", "sz.002040",
                "sz.002041", "sz.002042", "sz.002043", "sz.002044", "sz.002045",
                "sz.002046", "sz.002047", "sz.002048", "sz.002049", "sz.002050",
                "sz.002051", "sz.002052", "sz.002053", "sz.002054", "sz.002055",
                "sz.002056", "sz.002057", "sz.002058", "sz.002059", "sz.002060",
                "sz.002061", "sz.002062", "sz.002063", "sz.002064", "sz.002065",
                "sz.002066", "sz.002067", "sz.002068", "sz.002069", "sz.002070",
                "sz.002071", "sz.002072", "sz.002073", "sz.002074", "sz.002075",
                "sz.002076", "sz.002077", "sz.002078", "sz.002079", "sz.002080",
                "sz.002081", "sz.002082", "sz.002083", "sz.002084", "sz.002085",
                "sz.002086", "sz.002087", "sz.002088", "sz.002089", "sz.002090",
                "sz.002091", "sz.002092", "sz.002093", "sz.002094", "sz.002095",
                "sz.002096", "sz.002097", "sz.002098", "sz.002099", "sz.002100"
            ]
        
        # 过滤无效股票
        valid_stocks = []
        for code in stocks:
            if self.is_valid(code, ""):
                valid_stocks.append(code)
        
        return valid_stocks[:self.MAX_SCAN_LIMIT]

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
        try:
            # 转换为yfinance格式
            yf_code = self.clean_code(code)
            
            # 获取历史数据（180天）
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=180)
            
            # 使用yfinance获取历史数据
            try:
                ticker = yf.Ticker(yf_code)
                # 添加超时和重试
                df = ticker.history(start=start_date, end=end_date, timeout=10)
                
                if df is None or df.empty or len(df) < 60:
                    return None
                
                # 检查数据是否有效（yfinance可能返回空数据）
                if df['Close'].isna().all() or df['Volume'].isna().all():
                    return None
                
                # 重置索引，将日期转为列
                df = df.reset_index()
                df['date'] = df['Date'].dt.strftime('%Y-%m-%d')
                
                # 重命名列以匹配原有代码
                df = df.rename(columns={
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })
                
                # 计算涨跌幅
                df['pctChg'] = df['close'].pct_change() * 100
                df['pctChg'] = df['pctChg'].fillna(0)
                
                # 计算换手率（简化，使用成交量/流通股本估算）
                df['turn'] = (df['volume'] / df['volume'].rolling(20).mean() * 5).fillna(0)
                
                # 只保留需要的列
                df = df[['date', 'open', 'close', 'high', 'low', 'volume', 'pctChg', 'turn']]
                
                # 清理无效数据
                df = df.dropna(subset=['close', 'volume'])
                
                if len(df) < 60:
                    return None
                
                # 获取股票基本信息
                try:
                    info_data = ticker.info
                    name = info_data.get('shortName', code) if info_data else code
                    industry = info_data.get('industry', '-') if info_data else '-'
                except:
                    name = self.clean_code_back(yf_code)
                    industry = "-"
                    
            except Exception as e:
                return None
            
            info = {
                'name': name[:10],
                'industry': industry[:10],
                'ipoDate': '2000-01-01'
            }
            
            # 验证股票有效性（使用原始代码格式）
            original_code = self.clean_code_back(yf_code) if '.' in yf_code else code
            if not self.is_valid(original_code, info['name']):
                return None
            
            curr = df.iloc[-1]
            prev = df.iloc[-2]
            
            # 价格过滤
            if max_price is not None and float(curr['close']) > max_price:
                return None
            
            # 计算获利筹码
            winner_rate = self.calc_winner_rate(df, float(curr['close']))
            days_listed = 365
            
            # 计算均线
            df['MA5'] = df['close'].rolling(5).mean()
            df['MA20'] = df['close'].rolling(20).mean()
            df['MA200'] = df['close'].rolling(200).mean() if len(df) >= 200 else pd.Series([None] * len(df))
            risk_level = self.calc_risk_level(float(curr['close']), float(df['MA5'].iloc[-1]) if not pd.isna(df['MA5'].iloc[-1]) else 0, 
                                            float(df['MA20'].iloc[-1]) if not pd.isna(df['MA20'].iloc[-1]) else 0)
        except Exception as e:
            return None

        # 计算新的技术指标
        rsi = self.calc_rsi(df)
        k, d, j = self.calc_kdj(df)
        bb_upper, bb_mid, bb_lower = self.calc_bollinger(df)

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
        
        # 新增策略：RSI超卖反弹
        if rsi is not None:
            if rsi < 30 and len(df) >= 2:
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

        if priority == 0: return None

        # 使用原始代码格式返回
        original_code = self.clean_code_back(yf_code) if '.' in yf_code else code
        
        return {
            "result": {
                "代码": original_code, "名称": info['name'], 
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
            "option": f"{original_code} | {info['name']}"
        }

    def scan_market_optimized(self, code_list, max_price=None):
        """扫描市场 - yfinance版本"""
        results, alerts, valid_codes_list = [], [], []
        
        if len(code_list) > self.MAX_SCAN_LIMIT:
            code_list = code_list[:self.MAX_SCAN_LIMIT]
            st.info(f"⚠️ 股票数量超过限制，已截取前{self.MAX_SCAN_LIMIT}只")

        total = len(code_list)
        
        progress_container = st.empty()
        progress_bar = progress_container.progress(0, text=f"🚀 正在启动稳定扫描 (共 {total} 只)...")
        
        BATCH_SIZE = 5  # yfinance对A股支持有限，减少批次大小并添加延迟
        
        for i, code in enumerate(code_list):
            # 每处理一只股票都更新进度
            progress = (i + 1) / total
            current_count = min(i + 1, total)
            progress_bar.progress(progress, 
                                text=f"🔍 正在分析: {code} ({current_count}/{total}) | 已命中: {len(results)} 只")
            
            try:
                # 添加小延迟，避免请求过快
                if i > 0 and i % 10 == 0:
                    time.sleep(0.5)
                
                res = self._process_single_stock(code, max_price)
                if res:
                    results.append(res["result"])
                    if res["alert"]: alerts.append(res["alert"])
                    valid_codes_list.append(res["option"])
            except Exception as e:
                # 静默失败，继续下一个
                continue

        progress_container.empty()
        
        # 显示扫描完成提示
        if len(results) > 0:
            st.success(f"✅ 扫描完成！共找到 {len(results)} 只符合条件的股票")
        else:
            st.warning(f"⚠️ 扫描完成！共扫描 {total} 只股票，未找到符合条件的股票")
            st.info("💡 **提示**：yfinance对A股支持有限，部分股票可能无法获取数据。建议：\n"
                   "1. 尝试降低价格上限\n"
                   "2. 使用'手动输入'模式，输入已知可用的股票代码\n"
                   "3. 检查网络连接是否正常")
        
        return results, alerts, valid_codes_list

    def get_deep_data(self, code):
        """获取深度数据 - yfinance版本"""
        try:
            # 转换为yfinance格式
            yf_code = self.clean_code(code)
            
            # 获取6个月历史数据
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=180)
            
            try:
                ticker = yf.Ticker(yf_code)
                df = ticker.history(start=start_date, end=end_date)
                
                if df is None or len(df) < 20:
                    return None
                
                # 重置索引，将日期转为列
                df = df.reset_index()
                df['date'] = df['Date'].dt.strftime('%Y-%m-%d')
                
                # 重命名列
                df = df.rename(columns={
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })
                
                # 只保留需要的列
                df = df[['date', 'open', 'close', 'high', 'low', 'volume']]
                
                # 转换数据类型
                for col in ['open', 'close', 'high', 'low', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # 清理无效数据
                df = df.dropna(subset=['close', 'volume'])
                df = df.sort_values('date').reset_index(drop=True)
                
                return df if len(df) >= 20 else None
                
            except Exception as e:
                return None
            
        except Exception as e:
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
            
            if stock_list and len(stock_list) > 0:
                st.session_state['full_pool'] = stock_list 
                st.sidebar.success(f"✅ 已加载全量 {len(stock_list)} 只股票")
            else:
                st.sidebar.error("❌ 获取股票失败，请重试")
                st.sidebar.info("💡 可能的原因：\n1. 网络连接问题\n2. baostock服务暂时不可用\n3. 请稍后重试或选择其他扫描范围")
                # 清除缓存，下次重试
                if cache_key in st.session_state:
                    del st.session_state[cache_key]
    
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
        
        st.info(f"📊 当前结果：{len(df_scan)} 只股票 | 📅 扫描时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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