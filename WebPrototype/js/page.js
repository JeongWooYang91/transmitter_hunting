
$(function(){
	var app = new PIXI.Application(800, 800);
	document.body.appendChild(app.view);
	
	var graphics = new PIXI.Graphics();

	var background = new PIXI.Sprite.fromImage('/kaistmap.jpg');
	background.width = 800;
	background.height = 800;
	app.stage.addChild(background);
	app.stage.addChild(graphics);
	
		
	
	var socket = io();
	socket.on('init-client', function(serverData){
		//alert(serverData.message);
	});
	
	
	socket.on('position-update', function(serverData){
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
	});
	
	var userMode = '';
	$('#A').click(function(){
		userMode = 'A';
		socket.emit('select', {mode: 'A'});
		$('#A').addClass('btn-danger');
		$('#B').removeClass('btn-primary');
	});
	$('#B').click(function(){
		userMode = 'B';
		socket.emit('select', {mode: 'B'});
		$('#A').removeClass('btn-danger');
		$('#B').addClass('btn-primary');
	});
	
	$(document).on('keydown', function( event ) {
		var key = event.which || event.keyCode;
		if(key == 49)
		{
			userMode = 'A';
			socket.emit('select', {mode: 'A'});
			$('#A').addClass('btn-danger');
			$('#B').removeClass('btn-primary');
		}
		if(key == 50)
		{
			userMode = 'B';
			socket.emit('select', {mode: 'B'});
			$('#A').removeClass('btn-danger');
			$('#B').addClass('btn-primary');
		}
		if(userMode != 'A' && userMode != 'B') return;
		//console.log(key);
		if(key == '37' || key == 65	) //left
		{
		  socket.emit('client-left');
		}
		if(key == '38' || key == 87) //up
		{
		  socket.emit('client-up');
		}
		if(key == '39' || key == 68) //right
		{
		  socket.emit('client-right');
		}
		if(key == '40' || key == 83) //down
		{
		  socket.emit('client-down');
		}
	});
  
});
	
	/* var userMode = null;
  var started = false;
  var oldData = {};
  var newData = {};
  var mygraphics = {};
  var FPS = 30;
  var stage = new PIXI.Container();
  var renderer = PIXI.autoDetectRenderer(800, 600, {backgroundColor : 0x1099bb});
  var nickName = '';
  $('#myModal').modal({
    keyboard:false,
    show:true,
    backdrop:'static'
  });
  //$('#myModal').modal('show');
  $('#watch').click(function(e){
    $('#myModal').modal('hide');
    $('#viewport').append(renderer.view);
    requestAnimationFrame(animate);
    userMode = 'watch';
    nickName = $('#nickName').val();
    if(!nickName) nickName = 'unnamed';
    socket.emit('usermode', {mode: 'watch', name: nickName});
    started = true;
  });

  $('#attend').click(function(e){
    $('#myModal').modal('hide');
    $('#viewport').append(renderer.view);
    requestAnimationFrame(animate);
    userMode = 'attend';
    nickName = $('#nickName').val();
    socket.emit('usermode', {mode: 'attend', name: nickName});
    if(!nickName) nickName = 'unnamed';
    started = true;
  });
  
//  var renderer = PIXI.autoDetectRenderer(800, 600, {backgroundColor : 0x1099bb});
// $(window).width(), $(window).height()

  socket.on('init-client', function(serverData){
    FPS = serverData.fps;
  });

  socket.on('user-data', function(realData){
    var meid = realData.you;
    var serverData = realData.data;
    t = serverData[meid];
    if(t.team === 'A'){
      $('#you').addClass('label-danger');
    }
    else if(t.team === 'B'){
      $('#you').addClass('label-info');
    }
    else{
      $('#you').addClass('label-default');
    }
    $('#you').html('YOU: ' + t.name + '(' + t.team + ')');


    $('#userlist').html('');
    $('#userlist')
    .append($('<table>').addClass('table').addClass('table-condensed')
        .append($('<tr>')
          .append($('<td>')
            .html('Name'))
          .append($('<td>')
            .html('Mode'))
          .append($('<td>')
            .html('Team'))
          .append($('<td>')
            .html('Time'))));
    for(var k in serverData){
      var t = serverData[k];

      if(t.team === 'A'){
        $('#userlist')
        .append($('<table>').addClass('table').addClass('table-condensed')
          .append($('<tr>').addClass('danger')
            .append($('<td>')
              .html(serverData[k].name))
            .append($('<td>')
              .html(serverData[k].mode))
            .append($('<td>')
              .html(serverData[k].team))
            .append($('<td>')
              .html(serverData[k].time))));
      }
      else if(t.team === 'B'){
        $('#userlist')
        .append($('<table>').addClass('table').addClass('table-condensed')
          .append($('<tr>').addClass('info')
            .append($('<td>')
              .html(serverData[k].name))
            .append($('<td>')
              .html(serverData[k].mode))
            .append($('<td>')
              .html(serverData[k].team))
            .append($('<td>')
              .html(serverData[k].time))));
      }
      else{
        $('#userlist')
        .append($('<table>').addClass('table').addClass('table-condensed')
          .append($('<tr>').addClass('active')
            .append($('<td>')
              .html(serverData[k].name))
            .append($('<td>')
              .html(serverData[k].mode))
            .append($('<td>')
              .html(serverData[k].team))
            .append($('<td>')
              .html(serverData[k].time))));

      }
      
    }
  });

  socket.on('body-update', function(serverData){
    $('#txt').html('scoreA:' + serverData.scoreA + ' vs. scoreB:' + serverData.scoreB);
    oldData = newData;
    newData = serverData;
  });

  $(document).on('keyup', function( event ) {
    if(!started || userMode === 'watch') return;
    var key = event.which || event.keyCode;
    if(key == '32')
      socket.emit('client-pushed-button', '');
    if(key == '37') //left
    {
      socket.emit('client-left');
    }
    if(key == '39') //right
    {
      socket.emit('client-right');
    }
  });

  function animate(){
    updateBodyData();
    requestAnimationFrame(animate);
    renderer.render(stage);
  }

  function updateBodyData(){
    for(var i in newData){
      var d = newData[i];
      var o = oldData[i];
      if(!o){
        o = {x: 0,y: 0,rotation: 0};
      }
      var ngraphics = mygraphics[d.id];
      if(!ngraphics){
        ngraphics = new PIXI.Graphics();
        mygraphics[d.id] = ngraphics;
        ngraphics.lineStyle(1, 0xFFFFFF);

        if(d.team === 'A') ngraphics.beginFill(0xFFFF00);
        else ngraphics.beginFill(0xFF00FF);


        if(d.type === 'circle'){          
          ngraphics.drawCircle(0, 0, d.r);
        }
        else if(d.type === 'convex-polygon'){
          ngraphics.beginFill(0x00FFFF);
          var w = 300;
          var h = 300;
          ngraphics.drawRect(-w/2, -h/2, w, h);
          ngraphics.beginFill(0xFF0000);
          ngraphics.drawRect(-w/2, -h/2, w, 10);
          ngraphics.beginFill(0x0000FF);
          ngraphics.drawRect(-w/2, h/2-10, w, 10);
        }
        else{

        }
        stage.addChild(ngraphics);
      }
      //ngraphics.x = d.x;
      //ngraphics.y = d.y;
      TweenLite.fromTo(ngraphics, 1/FPS, 
        {x: d.x, y: d.y, rotation: d.ang}, 
        {x: o.x, y: o.y, rotation: o.ang});
    }
    oldData = newData;
  }
  */






