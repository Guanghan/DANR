connections=6
theads=6
duration=3600s
timeout=10s
address="http://"
port=80
api="imagereview"

wrk -c $connections -t $theads -d $duration --timeout=$timeout -s scripts/weapon.lua $address:$port/$api
