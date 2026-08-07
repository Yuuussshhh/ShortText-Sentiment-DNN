import pymysql
print("pymysql OK")
conn = pymysql.connect(
    host='localhost', user='root', password='ysh040822',
    database='sentiment_db', cursorclass=pymysql.cursors.DictCursor
)
cur = conn.cursor()
cur.execute('SELECT COUNT(*) as cnt FROM predictions')
row = cur.fetchone()
print(f'predictions table OK, {row["cnt"]} rows')
cur.execute("SELECT label, COUNT(*) as cnt FROM predictions GROUP BY label")
rows = cur.fetchall()
print(f'Stats: {rows}')
cur.close()
conn.close()
print("All OK!")