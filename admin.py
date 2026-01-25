import streamlit as st
import json
import os
import pandas as pd
from io import BytesIO
from datetime import datetime
import hashlib

# ==========================================
# 管理员配置
# ==========================================
ADMIN_PASSWORD = "admin2024"  # 管理员密码，建议修改为更安全的密码
USERS_FILE = "users.json"

# ==========================================
# 工具函数
# ==========================================
def hash_password(password):
    """使用SHA256哈希密码"""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    """加载用户数据"""
    try:
        # 获取当前工作目录和文件路径（用于调试）
        current_dir = os.getcwd()
        file_path = os.path.abspath(USERS_FILE)
        
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # 调试信息：如果文件不存在，尝试列出当前目录的文件
            try:
                files_in_dir = os.listdir('.')
            except:
                files_in_dir = []
    except Exception as e:
        pass
    return {}

def get_holdings_file(username):
    """根据用户名获取持仓文件路径"""
    safe_username = "".join(c for c in username if c.isalnum() or c in ('-', '_'))
    return f"holdings_data_{safe_username}.json"

def load_user_holdings(username):
    """加载指定用户的持仓数据"""
    try:
        holdings_file = get_holdings_file(username)
        if os.path.exists(holdings_file):
            with open(holdings_file, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        pass
    return []

# ==========================================
# 管理员登录
# ==========================================
def check_admin():
    """管理员登录验证"""
    if "admin_logged_in" not in st.session_state:
        st.markdown("### 🔐 管理员后台系统")
        st.warning("⚠️ 此页面仅供管理员使用")
        
        admin_pwd = st.text_input("请输入管理员密码", type="password", key="admin_password")
        
        if st.button("登录", type="primary"):
            # 验证管理员密码
            if admin_pwd == ADMIN_PASSWORD:
                st.session_state["admin_logged_in"] = True
                st.success("✅ 登录成功")
                st.rerun()
            else:
                st.error("❌ 密码错误")
        
        st.stop()
    return True

# ==========================================
# 主程序
# ==========================================
st.set_page_config(
    page_title="管理员后台",
    layout="wide",
    page_icon="👨‍💼",
    initial_sidebar_state="expanded"
)

if not check_admin():
    st.stop()

st.title("👨‍💼 管理员后台系统")
st.caption("用户数据管理与统计")

# ==========================================
# 侧边栏导航
# ==========================================
st.sidebar.header("📊 管理功能")
page = st.sidebar.radio(
    "选择功能",
    ["用户列表", "持仓详情", "数据统计", "数据导出", "🔧 调试信息"]
)

# ==========================================
# 1. 用户列表
# ==========================================
if page == "用户列表":
    st.header("👥 所有注册用户")
    
    users = load_users()
    
    if not users:
        st.info("📭 暂无注册用户")
    else:
        # 显示用户统计
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("总用户数", len(users))
        with col2:
            # 计算有持仓的用户数
            users_with_holdings = sum(1 for username in users.keys() if load_user_holdings(username))
            st.metric("有持仓用户", users_with_holdings)
        with col3:
            users_without_holdings = len(users) - users_with_holdings
            st.metric("无持仓用户", users_without_holdings)
        
        st.markdown("---")
        
        # 用户列表表格
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
            
            # 搜索功能
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

# ==========================================
# 2. 持仓详情
# ==========================================
elif page == "持仓详情":
    st.header("💼 用户持仓详情")
    
    users = load_users()
    
    if not users:
        st.info("📭 暂无注册用户")
    else:
        # 选择用户
        selected_user = st.selectbox(
            "选择要查看的用户",
            ["全部用户"] + list(users.keys())
        )
        
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
                
                # 统计信息
                st.markdown("### 📈 持仓统计")
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
                # 显示持仓详情
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
                
                # 计算总成本
                total_cost = sum(h.get('buy_price', 0) * h.get('quantity', 0) for h in holdings)
                st.metric("总持仓成本", f"¥{total_cost:,.2f}")

# ==========================================
# 3. 数据统计
# ==========================================
elif page == "数据统计":
    st.header("📊 数据统计")
    
    users = load_users()
    
    if not users:
        st.info("📭 暂无数据")
    else:
        # 收集所有持仓数据
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
            
            # 统计卡片
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
            
            # 热门股票排行
            st.subheader("🔥 热门股票排行（持有用户数）")
            stock_user_count = df_stats.groupby("股票代码")["用户名"].nunique().sort_values(ascending=False)
            if len(stock_user_count) > 0:
                df_popular = pd.DataFrame({
                    "股票代码": stock_user_count.index,
                    "持有用户数": stock_user_count.values
                })
                st.dataframe(df_popular.head(20), hide_index=True, use_container_width=True)
            
            st.markdown("---")
            
            # 持仓数量排行
            st.subheader("📈 持仓数量排行（总股数）")
            stock_quantity = df_stats.groupby("股票代码")["数量"].sum().sort_values(ascending=False)
            if len(stock_quantity) > 0:
                df_quantity = pd.DataFrame({
                    "股票代码": stock_quantity.index,
                    "总持股数": stock_quantity.values
                })
                st.dataframe(df_quantity.head(20), hide_index=True, use_container_width=True)
            
            st.markdown("---")
            
            # 用户持仓排行
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

# ==========================================
# 4. 数据导出
# ==========================================
elif page == "数据导出":
    st.header("📥 数据导出")
    
    users = load_users()
    
    if not users:
        st.info("📭 暂无数据可导出")
    else:
        # 导出选项
        export_type = st.radio(
            "选择导出类型",
            ["所有用户信息", "所有持仓数据", "统计数据"]
        )
        
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
                
                # 创建Excel
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
                
                # 创建Excel
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
                
                # 创建多个sheet的Excel
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    # 用户统计
                    user_stats = df_stats.groupby("用户名").size().reset_index(name="持仓数量")
                    user_stats.to_excel(writer, index=False, sheet_name='用户统计')
                    
                    # 股票统计
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

# ==========================================
# 5. 调试信息
# ==========================================
elif page == "🔧 调试信息":
    st.header("🔧 系统调试信息")
    
    # 显示当前工作目录
    st.subheader("📁 文件系统信息")
    current_dir = os.getcwd()
    st.code(f"当前工作目录: {current_dir}")
    
    # 显示文件路径
    users_file_path = os.path.abspath(USERS_FILE)
    st.code(f"users.json 路径: {users_file_path}")
    st.code(f"users.json 是否存在: {os.path.exists(USERS_FILE)}")
    
    # 列出当前目录的所有文件
    st.subheader("📋 当前目录文件列表")
    try:
        files = os.listdir('.')
        # 过滤出相关的JSON文件
        json_files = [f for f in files if f.endswith('.json')]
        data_files = [f for f in json_files if 'users' in f or 'holdings' in f]
        
        if data_files:
            st.success(f"✅ 找到 {len(data_files)} 个数据文件:")
            for file in sorted(data_files):
                file_path = os.path.abspath(file)
                file_size = os.path.getsize(file) if os.path.exists(file) else 0
                st.code(f"  - {file} ({file_size} bytes)")
        else:
            st.warning("⚠️ 未找到数据文件")
        
        # 显示所有文件（用于调试）
        with st.expander("查看所有文件"):
            st.code('\n'.join(sorted(files)))
    except Exception as e:
        st.error(f"❌ 无法列出文件: {str(e)}")
    
    # 尝试加载用户数据
    st.subheader("👥 用户数据加载测试")
    users = load_users()
    if users:
        st.success(f"✅ 成功加载 {len(users)} 个用户")
        st.json(users)
    else:
        st.warning("⚠️ 未加载到用户数据")
        # 尝试直接读取文件
        if os.path.exists(USERS_FILE):
            st.info("文件存在，尝试直接读取...")
            try:
                with open(USERS_FILE, 'r', encoding='utf-8') as f:
                    content = f.read()
                    st.code(f"文件内容长度: {len(content)} 字符")
                    if content:
                        try:
                            data = json.loads(content)
                            st.success("✅ 文件内容可以解析为JSON")
                            st.json(data)
                        except json.JSONDecodeError as e:
                            st.error(f"❌ JSON解析错误: {str(e)}")
                            st.code(content[:500])  # 显示前500个字符
            except Exception as e:
                st.error(f"❌ 读取文件失败: {str(e)}")
        else:
            st.error("❌ users.json 文件不存在")
    
    # 测试持仓文件
    st.subheader("💼 持仓文件测试")
    if users:
        for username in list(users.keys())[:5]:  # 只测试前5个用户
            holdings_file = get_holdings_file(username)
            exists = os.path.exists(holdings_file)
            st.code(f"{username}: {holdings_file} - {'存在' if exists else '不存在'}")
            if exists:
                holdings = load_user_holdings(username)
                st.code(f"  持仓数量: {len(holdings)}")

# ==========================================
# 退出登录
# ==========================================
st.sidebar.markdown("---")
if st.sidebar.button("🚪 退出登录"):
    st.session_state["admin_logged_in"] = False
    st.rerun()

st.sidebar.caption(f"👨‍💼 管理员后台 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

