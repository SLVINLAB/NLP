import cx_Oracle

db_session = cx_Oracle.connect('ufngdba/venus2002@10.10.1.50:1521/fndb2')
# m_gicode = "A005490"
# m_giname = "POSCO"
# m_gicode_noA= "005490"

m_gicode = "A064850"
m_giname = "에프앤가이드"
m_gicode_noA= "064850"

m_service_type = '1'
#  타입은    1 -  리포트,컨센서스 보유기업    [거래소]
#  타입은    2 -  리포트,컨센서스 미보유기업  [코스닥]
#  타입은    3 -  코넥스,프리보드             [코넥스,프리보드]
#  타입은    4 -  완전비상장                  [장외우량기업]
#  타입은 계속 추가가능함.
