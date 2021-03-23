var readline=require('readline')
var rl=readline.createInterface({
	input:process.stdin,
	output:process.stdout
});
rl.on('line',function(line){
	switch(line.trim()){
		case 'copy':
			console.log('复制');
			break;
		case 'hello':
			rl.write('write');
			console.log('world!');
			break;
		case 'close':
			rl.close();
			break;
		default:
			console.log('没有找到命令！');
			process.exit(0);
			break;
	}	
});
rl.on('close',function(){
	console.log('bye');
	process.exit(0);
});
