package com.example.admin.loginapp;

import android.*;
import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.location.Location;
import android.location.LocationListener;
import android.location.LocationManager;
import android.support.annotation.Keep;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.BaseExpandableListAdapter;
import android.widget.Button;
import android.widget.ExpandableListView;
import android.widget.TextView;
import android.widget.Toast;

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

import java.util.ArrayList;

public class GameActivity extends AppCompatActivity implements View.OnClickListener {

    private FirebaseAuth mAuth;
    private TextView stateText;
    private Button buttonJoin, buttonLogout;

    final private DatabaseReference
            mDatabase = FirebaseDatabase.getInstance().getReference("gameroom");
    private final static String TAG = "GameActivity";

    FirebaseUser user;
    boolean enableJoinButton = false;
    boolean buttonAvailable = false;
    String myUID = null;
    ValueEventListener fveListener;

    // GPS
    LocationManager locationManager;
    LocationListener locationListener;
    boolean isGPSEnabled, isNetworkEnabled;
    private static final int MY_PERMISSIONS_REQUEST_GPS = 1;

    // elistview
    private ArrayList<String> mGroupList = null;
    private ArrayList<ArrayList<String>> mChildList = null;
    private ArrayList<String> mFoxListContent = null;
    private ArrayList<String> mHoundListContent = null;
    private ExpandableListView mListView;
    private BaseExpandableListAdapter mBaseAdapter;

    private boolean foxExist = true;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_game);


        mAuth = mAuth.getInstance();
        if(mAuth.getCurrentUser()==null)
        {
            finish();
            startActivity(new Intent(this, MainActivity.class));
        }user = mAuth.getCurrentUser();


        Log.e(TAG, "Authorized anyway");

        stateText = (TextView) findViewById(R.id.textView);
        stateText.setText("Initializing ...");

        buttonLogout = (Button)findViewById(R.id.logout_button);
        buttonLogout.setOnClickListener(this);
        buttonJoin = (Button)findViewById(R.id.join_button);
        buttonJoin.setOnClickListener(this);


        mListView = (ExpandableListView) findViewById(R.id.elv_list);
        mGroupList = new ArrayList<String>();
        mChildList = new ArrayList<ArrayList<String>>();
        mFoxListContent = new ArrayList<String>();
        mHoundListContent = new ArrayList<String>();

        mGroupList.add("FOXES");
        mGroupList.add("HOUNDS");


        mChildList.add(mFoxListContent);
        mChildList.add(mHoundListContent);

        mBaseAdapter = new BaseExpandableAdapter(this, mGroupList, mChildList);
        mListView.setAdapter(mBaseAdapter);

        // 그룹 클릭 했을 경우 이벤트
        mListView.setOnGroupClickListener(new ExpandableListView.OnGroupClickListener() {
            @Override
            public boolean onGroupClick(ExpandableListView parent, View v,
                                        int groupPosition, long id) {
                //Toast.makeText(getApplicationContext(), "g click = " + groupPosition,
                        //Toast.LENGTH_SHORT).show();
                if(buttonAvailable){
                    if(!enableJoinButton){ // already joined state
                        if(groupPosition == 0){
                            mDatabase.child("members").child(myUID).child("role").setValue("fox");
                        }
                        else if(groupPosition == 1){
                            mDatabase.child("members").child(myUID).child("role").setValue("hound");
                        }

                    }
                }
                return true;
            }
        });

        // 차일드 클릭 했을 경우 이벤트
        mListView.setOnChildClickListener(new ExpandableListView.OnChildClickListener() {
            @Override
            public boolean onChildClick(ExpandableListView parent, View v,
                                        int groupPosition, int childPosition, long id) {
                //Toast.makeText(getApplicationContext(), "c click = " + childPosition,
                        //Toast.LENGTH_SHORT).show();
                return true;
            }
        });

        // 그룹이 닫힐 경우 이벤트
        mListView.setOnGroupCollapseListener(new ExpandableListView.OnGroupCollapseListener() {
            @Override
            public void onGroupCollapse(int groupPosition) {
                //Toast.makeText(getApplicationContext(), "g Collapse = " + groupPosition,
                        //Toast.LENGTH_SHORT).show();
            }
        });

        // 그룹이 열릴 경우 이벤트
        mListView.setOnGroupExpandListener(new ExpandableListView.OnGroupExpandListener() {
            @Override
            public void onGroupExpand(int groupPosition) {
                //Toast.makeText(getApplicationContext(), "g Expand = " + groupPosition,
                        //Toast.LENGTH_SHORT).show();

            }
        });

        // Attach a listener to read the data at our posts reference
        mDatabase.addValueEventListener(new ValueEventListener() {
            @Override
            public void onDataChange(DataSnapshot dataSnapshot) {
                buttonAvailable = false;
                enableJoinButton = true;
                mFoxListContent.clear();
                mHoundListContent.clear();
                foxExist = false;
                if(dataSnapshot.exists() && dataSnapshot.child("creator").exists() && dataSnapshot.child("state").exists()){
                    if(dataSnapshot.child("state").getValue().toString().equalsIgnoreCase("started")){
                        mDatabase.removeEventListener(this);
                        startActivity(new Intent(getBaseContext(), MapActivity.class));
                    }
                    else {
                        String txt = "";
                        Member member;
                        for (DataSnapshot ds : dataSnapshot.child("members").getChildren()){
                            member = ds.getValue(Member.class);
                            txt += "members: " + member.email + "[" + member.role + "]" + "\n\t(" + member.lat + ", " + member.lng + ")\n";

                            if(user.getEmail().equalsIgnoreCase(member.email)){
                                enableJoinButton = false;
                                myUID = ds.getKey().toString();
                            }
                            if(member.role.equalsIgnoreCase("fox")){
                                mFoxListContent.add(member.email);
                                foxExist = true;
                            }
                            if(member.role.equalsIgnoreCase("hound")){
                                mHoundListContent.add(member.email);
                            }

                        }

                        if(enableJoinButton){
                            buttonJoin.setText("Join");
                        }
                        else {
                            buttonJoin.setText("Exit");
                        }

                        /*
                        stateText.setText(
                                dataSnapshot.child("creator").getValue().toString() + "\n"
                                        + txt
                                        + dataSnapshot.child("members").getChildrenCount() + "\n"
                                        + dataSnapshot.child("state").getValue().toString()
                        );
                        */
                        Log.e(TAG, "Data loaded successfully." + dataSnapshot.getValue());

                        String myState = dataSnapshot.child("state").getValue().toString();
                        if(myState.equalsIgnoreCase("ready")){
                            buttonAvailable = true;
                            stateText.setText("Ready");
                        }
                        else if(myState.equalsIgnoreCase("starting") && dataSnapshot.child("time").exists()){
                            buttonAvailable = false;
                            stateText.setText("Game starts in " + dataSnapshot.child("time").getValue().toString() + " seconds...");
                        }
                        else {
                            buttonAvailable = false;
                        }

                    }
                }
                else {
                    stateText.setText("No game room found");
                    Log.e(TAG, "Data not exist.");
                }
                mBaseAdapter.notifyDataSetChanged();
            }

            @Override
            public void onCancelled(DatabaseError databaseError) {
                System.out.println("The read failed: " + databaseError.getCode());
            }
        });



    }

    /**
     * ATTENTION: This was auto-generated to implement the App Indexing API.
     * See https://g.co/AppIndexing/AndroidStudio for more information.
     */
    public Action getIndexApiAction() {
        return Actions.newView("Game", "http://[ENTER-YOUR-URL-HERE]");
    }

    @Override
    public void onStart() {
        super.onStart();

        // ATTENTION: This was auto-generated to implement the App Indexing API.
        // See https://g.co/AppIndexing/AndroidStudio for more information.
        FirebaseUserActions.getInstance().start(getIndexApiAction());
    }

    @Override
    public void onStop() {

        // ATTENTION: This was auto-generated to implement the App Indexing API.
        // See https://g.co/AppIndexing/AndroidStudio for more information.
        FirebaseUserActions.getInstance().end(getIndexApiAction());
        super.onStop();
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

    @Override
    public void onClick(View v) {
        if (v == buttonLogout)
        {
            if(myUID != null)
                mDatabase.child("members").child(myUID).removeValue();
            mAuth.signOut();
            finish();
            startActivity(new Intent(this, MainActivity.class));
        }
        if (v == buttonJoin)
        {
            if (buttonAvailable){
                if(enableJoinButton){
                    //synchronizeGPSLocation();
                    mDatabase.child("members").push().setValue(new Member(user.getEmail(), 0.0, 0.0, "hound"));
                }
                else {
                    mDatabase.child("members").child(myUID).removeValue();
                }
            }
        }
    }


}
