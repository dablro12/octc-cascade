# web/web.py 를 streamlit run 명령어로 실행시키는 스크립트 log를 남기기 위해 nohup을 사용
nohup streamlit run web/web.py > web/log/log.txt 2>&1 &
