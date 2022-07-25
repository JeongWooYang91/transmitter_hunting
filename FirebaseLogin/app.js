var express = require('express');
const exec = require('child_process').exec;

var admin = require("firebase-admin");
var serviceAccount = require(__dirname + "/cs408-transmitter-hunting-firebase-adminsdk-pnji9-9e850c0c69.json");
admin.initializeApp({
  credential: admin.credential.cert(serviceAccount),
  databaseURL: "https://cs408-transmitter-hunting.firebaseio.com"
});

var app = express();
app.use(express.static('public'));
app.get('/', function (req, res) {
	res.sendFile(__dirname + '/public/login.html');
});

app.listen(3000, function () {
  console.log('Example app listening on port 3000!');
});
available = true;
var db = admin.database();
var ref = db.ref("gameroom");
var calculate = false;
var members = null;
ref.child("members").on("value", function(snapshot) {
	members = snapshot.val();
});

var gx, gy;
var ax = [], ay = [], hid = [];
var signal;


var cx = 1273623389;
var cy = 363706170;

var px, py;
function calculateSignal(){
	console.log('calculate signal');
	var count = 0;
	var fox = 0;
	ax = [], ay = [], hid = [];
	for(var mid in members){
		var m = members[mid];
		//console.log(m);
		if(!m.email){
			ref.child("members").child(mid).remove();
		}
		else {
			if(m.lng && m.lat){
				px = parseInt(m.lng * 1e7);
				py = parseInt(m.lat * 1e7);
			}
			else {
				px = 0;
				py = 0;
			}
			
			if(m.role == 'fox'){
				gx = px;
				gy = py;
				fox++;
			}
			if(m.role == 'hound'){
				ax[mid]=px;
				ay[mid]=py;
				hid.push(mid);
				count++;
			}
		}
	}
	
	if(fox != 1 || count < 1){
		console.log("no member found");
		return;
	}
	signal = null;
	
	for(var i in hid){
		id = hid[i];
		if((ax[id] == 0 && ay[id] == 0) || (gx == ax[id] && gy == ay[id])){
			available = true;
		}
	}
	
	if (gx == 0 && gy == 0){
		available = true;
	}

	if (available == false) {
		var prog = "ServerCuda.exe " + gx + " " + gy + " " + hid.length;
		for (var i in hid){
			id = hid[i];
			prog += " " + ax[id] + " " + ay[id];
		}
		console.log(prog);
		exec(prog, (err, stdout, stderr) => {
			if (err) {
				console.error(err);
				return;
			}
			//console.log(stdout);
			signal = JSON.parse(stdout);
			//console.log(signal);
			
			//console.log(hid[0], signal);
			console.log("Newly calculated: ", (new Date()).toGMTString(), signal.length);
			for (var i in hid){
				id = hid[i];
				ref.child("members").child(id).child("signal").set(JSON.stringify(signal[i]));
			}
			available = true;
		});
	}
	
	
	
}

var timer;
var intervalID = null;
//starting the actual game
ref.child("state").on("value", function(snapshot){
	if(intervalID){
		clearInterval(intervalID);
		intervalID = null;
	}
	var val = snapshot.val();
	if(!val){
		return;
	}
	if(val == "starting"){
		timer = 5;
		ref.child("time").set(timer);
		intervalID = setInterval(function(){
			if(timer > 0) timer--;
			console.log("starting " + timer);
			
			ref.child("time").set(timer);
			if(timer == 0){
				ref.child("state").set("started");
			}
		}, 1000);
	}
	if(val == "started"){
		timer = 60*30;
		ref.child("time").set(timer);
		
		intervalID = setInterval(function(){
			if(timer%1 == 0 && available){
				available = false;
				console.log("calculating the signal");
				calculateSignal();
			}
			if(timer > 0) timer--;
			
			ref.child("time").set(timer);
			if(timer == 0){
				ref.child("state").set("finished");
			}
		}, 1000);
	}
});

