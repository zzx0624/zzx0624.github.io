var events=require('events');
var eventEmitter=new events.EventEmitter();
var listener1=function listenner1(){
	console.log('监听器listnner1执行')
}

var listener2=function listenner2(){
	console.log('监听器listnner2执行')
}
eventEmitter.addListener('connection',listener1);
eventEmitter.on('connection',listener2);
var eventListeners=eventEmitter.listenerCount('connection');
console.log(eventListeners+"个监听其监听连接事件");
eventEmitter.emit('connection');
eventEmitter.removeListener('connection',listener1);
console.log("listener1无了");
eventEmitter.emit('connection');
eventListeners=eventEmitter.listenerCount('connection');
console.log(eventListeners+"个监听");
console.log('程序执行完毕');
