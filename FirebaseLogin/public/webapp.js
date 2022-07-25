/*
 * Copyright 2016 Google Inc. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the
 * License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 * express or implied. See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * FirebaseUI initialization to be used in a Single Page application context.
 */

/**
 * @return {!Object} The FirebaseUI config.
 */
var audio = new Audio('audio_file.mp3');
audio.play();

function getUiConfig() {
  return {
    'callbacks': {
      // Called when the user has been successfully signed in.
      'signInSuccess': function(user, credential, redirectUrl) {
        handleSignedInUser(user);
        // Do not redirect.
        return false;
      }
    },
    // Opens IDP Providers sign-in flow in a popup.
    'signInFlow': 'popup',
    'signInOptions': [
      // TODO(developer): Remove the providers you don't need for your app.
      {
        provider: firebase.auth.GoogleAuthProvider.PROVIDER_ID,
        scopes: ['https://www.googleapis.com/auth/plus.login']
      },
      {
        provider: firebase.auth.EmailAuthProvider.PROVIDER_ID,
        // Whether the display name should be displayed in Sign Up page.
        requireDisplayName: false
      },
    ],
    // Terms of service url.
    'tosUrl': '/'
  };
}

// Initialize the FirebaseUI Widget using Firebase.
var ui = new firebaseui.auth.AuthUI(firebase.auth());

/**
 * Displays the UI for a signed in user.
 * @param {!firebase.User} user
 */
 

var cx = 1273623389;
var cy = 363706170;

var app = new PIXI.Application(800, 800);
//document.body.appendChild(app.view);

var graphics = new PIXI.Graphics();

var background = new PIXI.Sprite.fromImage('/kaistmap.jpg');
background.width = 800;
background.height = 800;
app.stage.addChild(background);
app.stage.addChild(graphics);

/*socket.on('position-update', function(serverData){
	//console.log(serverData);
	var pos = serverData.pos;
	graphics.clear();
	graphics.lineStyle(0);
	graphics.beginFill(0xFF0000, .5);
	graphics.drawCircle(pos.gx, pos.gy, 5);
	
	graphics.lineStyle(0);
	graphics.beginFill(0x0000FF, .5);
	graphics.drawCircle(pos.ax, pos.ay, 5);
	
	var sig = serverData.data;
	//console.log(sig);
	//alert(serverData.message);
	//graphics.position.set(pos.ax, pos.ay);
	
	for(var i = 0; i < sig.length; i++){
		//console.log(sig[i]);
		// Move it to the beginning of the line
		//graphics.position.set(pos.ax, pos.ay);
		
		if(sig[i] == 0) continue;
		var tx = -(sig[i + 360]) / sig[i] / 50;
		var ty = -(sig[i + 360*2]) / sig[i] / 50;
		//glVertex3f(pos.ax + tx, pos.ay + ty, 0.0f);
		
		//graphics.position.set(pos.ax, pos.ay);
		// Draw the line (endPoint should be relative to myGraph's position)
		//console.log(tx, ty);
		graphics.lineStyle(2, 0xff00ff)
			   .moveTo(pos.ax, pos.ay)
			   .lineTo(pos.ax + tx, pos.ay - ty);
	}
});*/

function tprint(time){
	var m = parseInt(time/60);
	var s = time%60;
	
	var str = '';
	if (m == 0) str += '00:';
	else if (m < 10) str += '0' + m + ':';
	else str += m + ':';
	
	if (s == 0) str += '00';
	else if (s < 10) str += '0' + s;
	else str += s;
	
	return str;
}

var gameroomRef = firebase.database().ref('/gameroom');

var handleSignedInUser = function(user) {
  document.getElementById('user-signed-in').style.display = 'block';
  document.getElementById('user-signed-out').style.display = 'none';
  document.getElementById('name').textContent = user.displayName;
  document.getElementById('email').textContent = user.email;
  document.getElementById('phone').textContent = user.phoneNumber;
  if (user.photoURL){
    document.getElementById('photo').src = user.photoURL;
    document.getElementById('photo').style.display = 'block';
  } else {
    document.getElementById('photo').style.display = 'none';
  }
  
					
    var timer;
    var intervalID;
  
  gameroomRef.on('value', function(snapshot){
		$("#menu-board").empty();
	    var val = snapshot.val();
		console.log(val);
		var fx, fy;
		if(val){
			var time = val.time;
			if(!val.creator){
				gameroomRef.set(null);
				return;
			}
			$("#menu-board").append("<p><b>Creator:</b> " + val.creator + " </p>")
			
			if(val.state == 'started') {
				graphics.clear();
				
				$("#menu-board").append($("<div>").append($("<b>").text(tprint(time))));
				
				$("#menu-board").append(app.view);
				
				for(var mid in val.members){
					var m = val.members[mid];
					var px, py;
					if(m.lng && m.lat){				
						px = (parseInt(m.lng * 1e7) - cx) * 400 / 80000 + 400;
						py = -(parseInt(m.lat * 1e7) - cy) * 400 / 80000 + 400;
					}
					else {
						px = 0;
						py = 0;
					}
					
					if(m.role == 'fox'){
						fx = px;
						fy = py;
					}
					if(m.role == 'hound'){
						if(m.signal){
							var sig = JSON.parse(m.signal);
							
							for (var i = 0; i < sig.length; i++) {
								var rad = i * (360 / sig.length) * 3.141592 / 180;
								var vx = Math.cos(rad);
								var vy = Math.sin(rad);
								var tx = sig[i]*(vx) / 1000;
								var ty = sig[i]*(vy) / 1000;
								console.log("drawed");
								graphics.lineStyle(1, 0xff00ff88)
									   .moveTo(px, py)
									   .lineTo(px + tx, py - ty);
							}
							
						}
						graphics.moveTo(0, 0);
						graphics.lineStyle(0);
						graphics.beginFill(0x0000FF, 1);
						graphics.drawCircle(px, py, 10);
						
					}
				}
				
				// draw fox now
				
				graphics.moveTo(0, 0);
				graphics.lineStyle(0);
				graphics.beginFill(0xFF0000, 1);
				graphics.drawCircle(fx, fy, 10);
			}
			else {				
				var ulm = $("<ul>").append($("<h4>").text("MEMBERS"));
				var ulf = $("<ul>").append($("<h4>").text("FOXES"));
				var ulh = $("<ul>").append($("<h4>").text("HOUNDS"));
				for(var mid in val.members){
					var m = val.members[mid];
					if(!m.email){
						gameroomRef.child("members").child(mid).remove();
					}
					else {
						ulm.append($("<li>").text(m.email + "[" + m.role + "]" + " (" + m.lat + ", " + m.lng + ")"));
						if(m.role == 'fox'){
							ulf.append($("<li>").text(m.email));
						}
						if(m.role == 'hound'){
							ulh.append($("<li>").text(m.email));
						}
					}
				}
				$("#menu-board").append(ulm);
				$("#menu-board").append(ulf);
				$("#menu-board").append(ulh);
				
				
				if(val.state == 'ready'){
					$("#menu-board").append($("<button>").attr("id", "start-game").text("Start game"));
					$("#start-game").click(function(){
						gameroomRef.child("state").set("starting");
					});
				}
				else if(val.state == 'starting'){
					if(time){
						$("#menu-board").append($("<div>").append($("<b>").attr("id", "starting-state").text("Game starts in " + time + " seconds...")));
					}
				}
			}
			
			$("#menu-board").append($("<br>"));
			$("#menu-board").append($("<button>").attr("id", "remove-room").text("Remove room"));
			$("#remove-room").click(function(){
				gameroomRef.set(null);
			});
		}
		else {
			$("#menu-board").append($("<button>").attr("id", "create-game").text("Create New Game"));
			$('#create-game').click(function(){
				gameroomRef.set({
					creator: firebase.auth().currentUser.email,
					state: 'ready'
					/*
					,
					members: {
						"-SAMPLEFOX": {
							email: "fox@test.com",
							lat: 36.3720970, // 363706170
							lng: 127.3600590, // 1273623389
							role: "fox"
						},
						"-SAMPLEHOUND": {
							email: "hound@test.com",
							lat: 36.3715570,
							lng: 127.3608189,
							role: "hound"
						},
						"-SAMPLEHOUND2": {
							email: "hound2@test.com",
							lat: 36.3733969,
							lng: 127.3613989,
							role: "hound"
						}
					}
					*/
				});
			});
		}
		
		
		
  });
};



/**
 * Displays the UI for a signed out user.
 */
var handleSignedOutUser = function() {
  document.getElementById('user-signed-in').style.display = 'none';
  document.getElementById('user-signed-out').style.display = 'block';
  ui.start('#firebaseui-container', getUiConfig());
};

// Listen to change in auth state so it displays the correct UI for when
// the user is signed in or not.
firebase.auth().onAuthStateChanged(function(user) {
  document.getElementById('loading').style.display = 'none';
  document.getElementById('loaded').style.display = 'block';
  user ? handleSignedInUser(user) : handleSignedOutUser();
});

/**
 * Deletes the user's account.
 */
var deleteAccount = function() {
  firebase.auth().currentUser.delete().catch(function(error) {
    if (error.code == 'auth/requires-recent-login') {
      // The user's credential is too old. She needs to sign in again.
      firebase.auth().signOut().then(function() {
        // The timeout allows the message to be displayed after the UI has
        // changed to the signed out state.
        setTimeout(function() {
          alert('Please sign in again to delete your account.');
        }, 1);
      });
    }
  });
};


/**
 * Initializes the app.
 */
var initApp = function() {
  document.getElementById('sign-out').addEventListener('click', function() {
    firebase.auth().signOut();
  });
  document.getElementById('delete-account').addEventListener(
      'click', function() {
        deleteAccount();
      });
};

window.addEventListener('load', initApp);
