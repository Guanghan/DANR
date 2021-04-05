wrk.method = "POST"
wrk.headers["content-type"] = "application/json"
math.randomseed(os.time())
number = math.random()
-- wrk.body = string.format('{"eature_type": "multigrain", "data": [{"data_id": "1", "content":"http://IA_20200724_2%03d.jpg"}] }', math.random(1, 100))

request = function()
   url = string.format("http://IA_20200724_2%03d.jpg", math.random(1, 100))
   -- print(url)
   local body = '{"request_no": "1", "business_id": "12", "data": [{"data_id": "1", "content":"' .. url .. '"}], "retrieve_source":[{"dataset_name":"default", "violation_type":{"p20":["p203"]}}]}'
   -- print(body)
   return wrk.format(wrk.method, wrk.path, wrk.headers, body)
end


function response(status, headers, body)
   -- print(string.sub(body, 0, 50))
   print(body)
end

done = function(summary, latency, requests)
   io.write("------------------------------\n")
   for _, p in pairs({ 50, 90, 99, 99.999 }) do
      n = latency:percentile(p)
      io.write(string.format("%g%%: %d\n", p, n))
   end
end
