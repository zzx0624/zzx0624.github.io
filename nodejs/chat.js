var net =require('net');
var clientList={};
var count=1;
var server=net.createServer(function (socket){
    clientList[count]=socket;
    console.log(count);
    socket.write('玩家'+count+'成功进入群聊！\r\n');
    count++;
    socket.on('data',function(data){
        console.log(data.toString());
        broadcast(data);
    });
    socket.on('end',function(){
        var x;
        for(var i=1;i<count;i++){
            if (clientList[i]==socket){
                x=i;
                delete clientList[i];
            }
        }
        broadcast('玩家'+x+'退出群聊！');
    });
});
function broadcast(data){
    for(var key in clientList){
        clientList[key].write(data);
    }

}
server.listen(10000,'127.0.0.1');
