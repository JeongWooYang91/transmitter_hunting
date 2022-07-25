var express = require('express');
var app = express();
var http = require('http').Server(app);
var io = require('socket.io')(http);

var position = {};
position.gx = 300;
position.gy = 300;
position.ax = 400;
position.ay = 400;

const exec = require('child_process').exec;

var signal = null;
function runCalculation(){
	var prog = "SeqProgram.exe " + position.gx + " " + position.gy + " " + position.ax + " " + position.ay;
	exec(prog, (err, stdout, stderr) => {
		if (err) {
			console.error(err);
			return;
		}
		//console.log(stdout);
		signal = JSON.parse(stdout);
		//console.log(signal);
	});
}
runCalculation(); //initalize


http.listen(3000, function(){
	console.log('listening on *:3000');
});

app.use(express.static(__dirname));
app.get('/', function(req, res){res.sendFile(__dirname + '/index.html');});
app.get('/kaistmap.jpg', function(req, res){res.sendFile(__dirname + '/kaistmap.jpg');});


var userData = {};
var playerNum = 0;
/*** Socket.io IO-CONNECTION ***/
io.on('connection', function(socket){
	socket.emit('init-client', {message: socket.id + " is connected"});
	io.sockets.connected[socket.id].emit('position-update', {pos: position, data: signal});
	userData[socket.id] = '';
	
	socket.on('select', function(data){
		console.log(data);
		userData[socket.id] = data.mode;
	});
	
	socket.on('disconnect', function(){
		console.log('user disconnected');
		delete userData[socket.id];
		io.emit('user-data', userData);
	});
	
	socket.on('client-left', function(){
		console.log('left');
		if(userData[socket.id]== 'A') position.gx -= 5;
		if(userData[socket.id]== 'B') position.ax -= 5;
		runCalculation();
		io.emit('position-update', {pos: position, data: signal});
	});
	socket.on('client-right', function(){
		console.log('right');
		if(userData[socket.id]== 'A') position.gx += 5;
		if(userData[socket.id]== 'B') position.ax += 5;
		runCalculation();
		io.emit('position-update', {pos: position, data: signal});
	});
	socket.on('client-up', function(){
		console.log('up');
		if(userData[socket.id]== 'A') position.gy -= 5;
		if(userData[socket.id]== 'B') position.ay -= 5;
		runCalculation();
		io.emit('position-update', {pos: position, data: signal});
	});
	socket.on('client-down', function(){
		console.log('down');
		if(userData[socket.id]== 'A') position.gy += 5;
		if(userData[socket.id]== 'B') position.ay += 5;
		runCalculation();
		io.emit('position-update', {pos: position, data: signal});
	});
});
