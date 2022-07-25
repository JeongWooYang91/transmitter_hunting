package com.example.admin.loginapp;

import android.*;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.location.Location;
import android.location.LocationListener;
import android.location.LocationManager;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.Chronometer;
import android.widget.ImageView;
import android.widget.RadioButton;
import android.widget.TextView;

import com.google.firebase.appindexing.Action;
import com.google.firebase.appindexing.FirebaseUserActions;
import com.google.firebase.appindexing.builders.Actions;
import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.auth.FirebaseUser;
import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;

import org.json.JSONArray;
import org.json.JSONException;


public class MapActivity extends AppCompatActivity implements View.OnClickListener {
    private final static String TAG = "MapActivity";
    private final static double move_unit = 0.0001;
    private static final int MY_PERMISSIONS_REQUEST_GPS = 1;

    final private DatabaseReference
            mDatabase = FirebaseDatabase.getInstance().getReference("gameroom");
    ImageView imgView;
    Button N1Button, E3Button, KISTIButton;
    Button upButton, downButton, leftButton, rightButton;
    RadioButton GPSButton, DEMOButton;

    TextView timer;

    // GPS
    LocationManager locationManager;
    LocationListener locationListener;
    private FirebaseAuth mAuth;
    FirebaseUser user;
    ValueEventListener fveListener;

    String myUID = null;
    double signal[] = new double[360];
    double mx, my, gpsx, gpsy;

    boolean IamFox = false;

    double lat, lng;
    boolean GPSMode = false;
    final static double mapx = 127.3623389;
    final static double mapy = 36.3706170;
    Bitmap myMap;
    Bitmap backBit;
    Bitmap getMinimap(){
        Bitmap result;
        int ppx = (int)((mx - mapx) * 1e7 + 80000);
        int ppy = (int)((my - mapy) * 1e7 + 80000);

        int width = backBit.getWidth();
        //int height = backBit.getHeight();
        int cropwidth = width / 20;
        int cropheight = cropwidth * 3 / 4;

        if (0 < ppx && ppx < 160000 && 0 < ppy && ppy < 160000) {

            int px = ppx * myMap.getWidth() / 160000;
            int py = ppy * myMap.getHeight() / 160000;


            int margin = myMap.getWidth() / 2;
            result = Bitmap.createBitmap(backBit, px - cropwidth + margin, backBit.getHeight() - (py + margin) - cropheight, cropwidth * 2, cropheight * 2);

            Canvas ncanvans = new Canvas(result);
            Paint Pnt = new Paint();

            float ax = ncanvans.getWidth()/2;
            float ay = ncanvans.getHeight()/2;
            Pnt.setColor(Color.CYAN);
            for(int i = 0; i < 360; i++){
                double length = signal[i];
                double rad = i * Math.PI / 180;
                float tx = (float) (Math.cos(rad) * length / 500.0);
                float ty = (float) (Math.sin(rad) * length / 500.0);
                ncanvans.drawLine(ax,ay,ax+tx,ay-ty,Pnt);
            }

            if (IamFox){
                Pnt.setColor(Color.RED);
            }
            else{
                Pnt.setColor(Color.BLUE);
            }

            ncanvans.drawCircle(ax,ay,10,Pnt);

            //backBit = Bitmap.createScaledBitmap(backBit, width, width * 3 / 4, true);
        }
        else {
            result = Bitmap.createBitmap(backBit, 0, 0, cropwidth * 2, cropheight * 2);
        }

        return result;
    }

    String tprint(int time){
        int m = time/60;
        int s = time%60;

        String str = "";
        if (m == 0) str += "00:";
        else if (m < 10) str += "0" + m + ":";
        else str += m + ":";

        Log.e(TAG, "TIME to calculate = " + time + " == "  +m + " : " + s + " == " + str);
        if (s == 0) str += "00";
        else if (s < 10) str += "0" + s;
        else str += s;

        Log.e(TAG, "TIME to calculate = " + time + " == "  +m + " : " + s + " == " + str);
        return str;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_map);

        myMap = BitmapFactory.decodeResource(getResources(), R.drawable.kaistmap);
        backBit = Bitmap.createBitmap(myMap.getWidth() * 2, myMap.getHeight() * 2, Bitmap.Config.ARGB_8888);

        Canvas canvas = new Canvas(backBit);
        canvas.drawARGB(255, 225, 225, 255);
        canvas.drawBitmap(myMap, myMap.getWidth() / 2, myMap.getHeight() / 2, null);

        imgView = (ImageView) findViewById(R.id.mapImageView);
        N1Button = (Button) findViewById(R.id.N1);
        E3Button = (Button) findViewById(R.id.E3);
        KISTIButton = (Button) findViewById(R.id.KISTI);
        upButton = (Button) findViewById(R.id.UP);
        downButton = (Button) findViewById(R.id.DOWN);
        leftButton = (Button) findViewById(R.id.LEFT);
        rightButton = (Button) findViewById(R.id.RIGHT);
        GPSButton = (RadioButton) findViewById(R.id.GPS);
        DEMOButton = (RadioButton) findViewById(R.id.DEMO);
        timer = (TextView) findViewById(R.id.TIMER);


        N1Button.setOnClickListener(this);
        E3Button.setOnClickListener(this);
        KISTIButton.setOnClickListener(this);
        upButton.setOnClickListener(this);
        downButton.setOnClickListener(this);
        leftButton.setOnClickListener(this);
        rightButton.setOnClickListener(this);
        GPSButton.setOnClickListener(this);
        DEMOButton.setOnClickListener(this);

        mx = 127.3645189;
        my = 36.3741570;

        for(int i = 0; i < 360; i++) {
            signal[i] = 0.0;
        }

        imgView.setImageBitmap(getMinimap());



        mAuth = mAuth.getInstance();
        if(mAuth.getCurrentUser()==null)
        {
            finish();
            startActivity(new Intent(this, MainActivity.class));
        }user = mAuth.getCurrentUser();

        DEMOButton.setChecked(true);

        // Attach a listener to read the data at our posts reference
        mDatabase.addValueEventListener(new ValueEventListener() {
            @Override
            public void onDataChange(DataSnapshot dataSnapshot) {
                if(!dataSnapshot.exists() || !dataSnapshot.child("state").getValue().toString().equalsIgnoreCase("started")){
                    mDatabase.removeEventListener(this);
                    startActivity(new Intent(getBaseContext(), GameActivity.class));
                }
                else {
                    if(!dataSnapshot.exists()){
                        return;
                    }

                    if(dataSnapshot.child("time").exists()){
                        Log.e(TAG, "TIME = " + dataSnapshot.child("time").getValue());
                        timer.setText(tprint(Integer.parseInt(dataSnapshot.child("time").getValue().toString())));
                    }
                    Member member;
                    if(dataSnapshot.child("members").exists()){
                        for (DataSnapshot ds : dataSnapshot.child("members").getChildren()){
                            member = ds.getValue(Member.class);

                            Log.e(TAG, "EQUAL = " + user.getEmail() + " ~~~~ " + member.email);
                            if(user.getEmail().equalsIgnoreCase(member.email)){
                                myUID = ds.getKey().toString();

                            }
                        }
                    }

                    if (dataSnapshot.child("members").child(myUID).exists()) {
                        mDatabase.child("members").child(myUID).child("lng").setValue(mx);
                        mDatabase.child("members").child(myUID).child("lat").setValue(my);

                        if(dataSnapshot.child("members").child(myUID).child("signal").exists()){
                            String sigStr = dataSnapshot.child("members").child(myUID).child("signal").getValue().toString();
                            try {
                                JSONArray sigData = new JSONArray(sigStr);
                                for(int i = 0; i < 360; i++) {
                                    signal[i] = Double.parseDouble(sigData.get(i).toString());
                                }
                            } catch (JSONException e) {
                                e.printStackTrace();
                            }
                        }
                        else {
                            IamFox = true;
                        }

                        if(dataSnapshot.child("members").child(myUID).child("role").exists()){
                            String roleStr = dataSnapshot.child("members").child(myUID).child("role").getValue().toString();
                            if(roleStr.equalsIgnoreCase("fox")){
                                IamFox = true;
                            }
                            else {
                                IamFox = false;
                            }
                        }
                    }

                    imgView.setImageBitmap(getMinimap());
                }
            }

            @Override
            public void onCancelled(DatabaseError databaseError) {
            }
        });


        locationManager = (LocationManager) this.getSystemService(Context.LOCATION_SERVICE);
        locationListener = new LocationListener() {
            public void onLocationChanged(Location location) {
                gpsy = location.getLatitude();
                gpsx = location.getLongitude();

                if (myUID != null && GPSMode) {
                    mx = gpsx;
                    my = gpsy;
                    mDatabase.child("members").child(myUID).child("lat").setValue(my);
                    mDatabase.child("members").child(myUID).child("lng").setValue(mx);
                }
            }

            public void onStatusChanged(String provider, int status, Bundle extras) {
                Log.w(TAG, "onStatusChanged");
            }

            public void onProviderEnabled(String provider) {
                Log.w(TAG, "onProviderEnabled");
            }

            public void onProviderDisabled(String provider) {
                Log.w(TAG, "onProviderDisabled");
            }
        };

        registerLocationUpdates();

    }



    private void registerLocationUpdates() {
        // Register the listener with the Location Manager to receive location updates
        if (ActivityCompat.checkSelfPermission(getBaseContext(), android.Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED
                && ActivityCompat.checkSelfPermission(getBaseContext(), android.Manifest.permission.ACCESS_COARSE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            // 권한 획득에 대한 설명 보여주기
            if (ActivityCompat.shouldShowRequestPermissionRationale(this,
                    android.Manifest.permission.ACCESS_FINE_LOCATION)) {

                // 사용자에게 권한 획득에 대한 설명을 보여준 후 권한 요청을 수행

            } else if (ActivityCompat.shouldShowRequestPermissionRationale(this,
                    android.Manifest.permission.ACCESS_COARSE_LOCATION)) {

                // 사용자에게 권한 획득에 대한 설명을 보여준 후 권한 요청을 수행

            }
            else {

                // 권한 획득의 필요성을 설명할 필요가 없을 때는 아래 코드를
                //수행해서 권한 획득 여부를 요청한다.

                ActivityCompat.requestPermissions(this,
                        new String[]{ android.Manifest.permission.ACCESS_FINE_LOCATION, android.Manifest.permission.ACCESS_COARSE_LOCATION},
                        MY_PERMISSIONS_REQUEST_GPS);

            }
        }
        locationManager.requestLocationUpdates(LocationManager.NETWORK_PROVIDER,
                1000, 1, locationListener);
        locationManager.requestLocationUpdates(LocationManager.GPS_PROVIDER,
                1000, 1, locationListener);
        //1000은 1초마다, 1은 1미터마다 해당 값을 갱신한다는 뜻으로, 딜레이마다 호출하기도 하지만
        //위치값을 판별하여 일정 미터단위 움직임이 발생 했을 때에도 리스너를 호출 할 수 있다.
    }


    @Override
    public void onClick(View v) {
        // Set location
        if(!GPSMode){
            if (v == N1Button)
            {
                mx = 127.3613989;
                my = 36.3733969;
            }
            if (v == E3Button)
            {
                mx = 127.3608189;
                my = 36.371557;
            }
            if (v == KISTIButton)
            {
                mx = 127.36089889999998;
            my = 36.36979689999988;
        }

        //
            if (v == upButton)
            {
                my += move_unit;
            }
            if (v == downButton)
            {
                my -= move_unit;
            }
            if (v == leftButton)
            {
                mx -= move_unit;
            }
            if (v == rightButton)
            {
                mx += move_unit;
            }
        }


        //
        if (v == GPSButton)
        {
            GPSMode = true;
            GPSButton.setChecked(true);
            DEMOButton.setChecked(false);
            mx = gpsx;
            my = gpsy;
        }
        if (v == DEMOButton)
        {
            GPSMode = false;
            GPSButton.setChecked(false);
            DEMOButton.setChecked(true);
        }


        if(myUID != null) {
            Log.e(TAG, "Lng = " + mx + "(" + gpsx + ") Lat = " + my + "(" + gpsy + ")");
            mDatabase.child("members").child(myUID).child("lng").setValue(mx);
            mDatabase.child("members").child(myUID).child("lat").setValue(my);
        }

        imgView.setImageBitmap(getMinimap());
    }


    public static final class Member{
        public String email;
        public double lat;
        public double lng;
        public String role;

        public Member(){}
        public Member(String email, double lat, double lng, String role){
            this.email = email;
            this.lat = lat;
            this.lng = lng;
            this.role = role;
        }
    }
}
